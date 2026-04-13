from typing import TypedDict, List, Dict, Any
import math
from copy import deepcopy
import os
from pathlib import Path

from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

from config import MIN_STEP_S, MAX_STEP_S, MODEL, ARRIVE_M, MAX_DISTANCE_M, SYSTEM_PROMPT, MAX_HISTORY, MAX_SAFE_SPEED_MPS
from krpc_client import KRPCContext
from state import StateTracker
from executor import Executor
from plotter import Plotter
from logger import Logger

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))

def _drop_orphan_tool_messages(msgs: list[Any]) -> list[Any]:
    cleaned: list[Any] = []
    for m in msgs:
        if isinstance(m, ToolMessage):
            if not cleaned:
                continue
            prev = cleaned[-1]
            if not (isinstance(prev, AIMessage) and prev.tool_calls):
                continue
            valid_ids = {tc["id"] for tc in prev.tool_calls}
            if m.tool_call_id not in valid_ids:
                continue
        cleaned.append(m)
    return cleaned

def read_target_from_csv(line_number: int, csv_path: str = "experiment_locations.csv") -> tuple[float, float]:
    """Read lat/lon from a specific line in the CSV file.
    
    Args:
        line_number: Line number to read (1-based, excluding header)
        csv_path: Path to the CSV file (relative to this script's directory)
        
    Returns:
        Tuple of (latitude, longitude)
    """
    import csv
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    full_path = script_dir / csv_path
    
    with open(full_path, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            if i == line_number:
                return (float(row['lat']), float(row['lon']))
    raise ValueError(f"Line {line_number} not found in {csv_path}")

def tools_factory(executor):

    @tool
    def drive(throttle: float, steer: float, duration_s: float) -> str:
        """throttle, steer in [-1,1]. duration_s in [MIN_STEP_S, MAX_STEP_S]."""
        throttle = _clamp(float(throttle), -1.0, 1.0)
        steer = _clamp(float(steer), -1.0, 1.0)
        duration_s = _clamp(float(duration_s), MIN_STEP_S, MAX_STEP_S)
        return executor.drive(throttle, steer, duration_s)

    @tool
    def brake(brake_strength: float, duration_s: float) -> str:
        """brake_strength in [0,100]. duration_s in [MIN_STEP_S, MAX_STEP_S]."""
        brake_strength = _clamp(float(brake_strength), 0.0, 100.0)
        duration_s = _clamp(float(duration_s), MIN_STEP_S, MAX_STEP_S)
        return executor.brake(brake_strength, duration_s)


    return [drive, brake]


class AgentState(TypedDict):
    messages: List[Any]          # chat history
    rover_state: Dict[str, Any]  # output of StateTracker.get_state
    target_latlon: Any
    step: int
    overridden_by_safety: bool   # flag indicating if last tool call was overridden



def llm_node_factory(llm):
    def llm_node(state: AgentState) -> AgentState:
        rover_state = state["rover_state"]

        print(f"distance_meters={rover_state['distance_meters']:.2f}m, forward_distance_meters={rover_state['forward_distance_meters']:.2f}m, right_distance_meters={rover_state['right_distance_meters']:.2f}m, speed_mps={rover_state['speed_mps']:.2f}m/s, bearing_error_deg={rover_state['bearing_error_deg']:.1f}°")
     

        observation = (
            f"OBSERVATION:\n"
            f"distance_meters={rover_state['distance_meters']:.2f}\n"
            f"forward_distance_meters={rover_state['forward_distance_meters']:+.2f}\n"
            f"right_distance_meters={rover_state['right_distance_meters']:+.2f}\n"
            f"speed_mps={rover_state['speed_mps']:.2f}\n"
            f"bearing_error_deg={rover_state['bearing_error_deg']:+.1f}\n"            
        )

        messages = list(state["messages"])
        
        messages.append(HumanMessage(content=observation))
        
        # truncate history if too long, keeping system prompt + last MAX_HISTORY messages
        if len(messages) > 1 + MAX_HISTORY:
            messages = [messages[0]] + messages[-MAX_HISTORY:]
            messages = _drop_orphan_tool_messages(messages)
            
            # Remove any leading ToolMessages after truncation to avoid orphaned tool responses
            while len(messages) > 1 and isinstance(messages[1], ToolMessage):
                messages.pop(1)

        response = llm.invoke(messages)
        return {**state, "messages": messages + [response]}
    return llm_node


def tool_node_factory(tools, executor, logger):
    tool_map = {t.name: t for t in tools}

    def tool_node(state: AgentState) -> AgentState:
        messages = list(state["messages"])
        last_message = messages[-1]        

        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:            
            status = executor.execute_action(
                {"throttle": 0.0, "steering": 0.0, "brake_strength": 100, "brakes": True},
                MIN_STEP_S,
            )
            messages.append(AIMessage(content=f"(fallback) no_tool_call => braked | {status}"))
            return {**state, "messages": messages, "step": state["step"] + 1}

        # Only execute the first tool call, but respond to all
        for i, tool_call in enumerate(last_message.tool_calls):
            name = tool_call["name"]
            args = tool_call["args"]

            tool_fn = tool_map.get(name)

            if tool_fn is None:
                # unknown tool => brake
                status = executor.execute_action(
                    {"throttle": 0.0, "steering": 0.0, "brake_strength": 100, "brakes": True},
                    MIN_STEP_S,
                )
                messages.append(ToolMessage(content=f"unknown_tool({name}) => braked | {status}", tool_call_id=tool_call["id"]))
            elif i == 0:
                # Execute only the first tool call
                result = tool_fn.invoke(args)
                logger.log(
                    step=state["step"],
                    rover_state=state["rover_state"],
                    tool_name=name,
                    args=args,
                    overridden=state.get("overridden_by_safety", False)
                )

                print(f"Executed tool call: {name}({args}) => {result}")
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
            else:
                # Respond to other calls without executing
                messages.append(ToolMessage(content=f"skipped (only first tool call allowed)", tool_call_id=tool_call["id"]))

        
        return {**state, "messages": messages, "step": state["step"] + 1}

    return tool_node

def _safe_float(x, default=0.0):
    try:
        v = float(x)
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default


def safety_node_factory():
    def safety_node(state: AgentState) -> AgentState:
        messages = list(state["messages"])
        rover_state = state["rover_state"]
        overridden = False

        last = messages[-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {**state, "overridden_by_safety": False}

        safe_last = deepcopy(last)
        
        tc = safe_last.tool_calls[0]
        name = tc["name"]

        speed = _safe_float(rover_state.get("speed_mps"), 0.0)

        if name == "drive":            
            if speed > MAX_SAFE_SPEED_MPS:
                print(f"⚠️  SAFETY OVERRIDE: speed {speed:.2f} m/s > {MAX_SAFE_SPEED_MPS} m/s, overriding drive with brake")
                tc["name"] = "brake"
                tc["args"] = {"brake_strength": 15.0, "duration_s": 1.0}
                overridden = True

        messages[-1] = safe_last
        return {**state, "messages": messages, "overridden_by_safety": overridden}

    return safety_node


def observe_node_factory(tracker, vessel, sc):
    def observe_node(state: AgentState) -> AgentState:
        new_rover_state = tracker.get_state(vessel, state["target_latlon"], sc)
        print(f"Step {state['step']:03d}: distance_meters={new_rover_state['distance_meters']:.2f}m, speed_mps={new_rover_state['speed_mps']:.2f}m/s, bearing_error_deg={new_rover_state['bearing_error_deg']:.1f}°")
        return {**state, "rover_state": new_rover_state, "overridden_by_safety": False}
    return observe_node

def update_plot_node_factory(plotter):
    def update_plot_node(state: AgentState) -> AgentState:
        rover_state = state["rover_state"]
        plotter.update_plot((rover_state["vessel_latitude_deg"], rover_state["vessel_longitude_deg"]))        
        return state
    return update_plot_node

def route(state: AgentState) -> str:
    s = state["rover_state"]
    # stop condition
    if s["distance_meters"] <= ARRIVE_M:
        return "end"

    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    
    return "tools"

def build_graph(executor, tracker, vessel, sc, plotter, logger):
    tools = tools_factory(executor)

    llm = ChatOpenAI(model=MODEL, temperature=0).bind_tools(tools, tool_choice="required")

    llm_node = llm_node_factory(llm)
    safety_node = safety_node_factory()
    tools_node = tool_node_factory(tools, executor, logger)
    observe_node = observe_node_factory(tracker, vessel, sc)
    update_plot_node = update_plot_node_factory(plotter)

    g = StateGraph(AgentState)
    g.add_node("observe", observe_node)
    g.add_node("llm", llm_node)
    g.add_node("safety", safety_node)
    g.add_node("tools", tools_node)
    g.add_node("update_plot", update_plot_node)

    g.set_entry_point("observe")
    g.add_edge("observe", "llm")
    g.add_conditional_edges("llm", route, {"tools": "safety", "end": END})
    g.add_edge("safety", "tools")
    g.add_edge("tools", "update_plot")
    g.add_edge("update_plot", "observe")

    return g.compile()


def main():

    for i in range(40):
        step = i+1
        krpc_context = KRPCContext()
        vessel = krpc_context.vessel
        space_center = krpc_context.sc
        tracker = StateTracker()
        plotter = Plotter()
        logger = Logger(f"{step}.csv")   
        space_center.load("quicksave")  
    
        # Uncomment this line to use a random target:
        # target_latlon = tracker.generate_random_target_latlon(vessel, space_center, MAX_DISTANCE_M)
        
        # Uncomment to read target from experiment_locations.csv for reproducibility.
        target_latlon = read_target_from_csv(line_number=step, csv_path="experiment_locations.csv")

        executor = Executor(krpc_context.ctrl, vessel, target_latlon, space_center, tracker)

        rover_state = tracker.get_state(vessel, target_latlon, space_center)

        plotter.setup_plot(target_latlon, (rover_state["vessel_latitude_deg"], rover_state["vessel_longitude_deg"]))

        app = build_graph(executor, tracker, vessel, space_center, plotter, logger)

        init_state = {
            "messages": [SystemMessage(content=SYSTEM_PROMPT)],
            "rover_state": rover_state,
            "target_latlon": target_latlon,
            "step": 0,
            "overridden_by_safety": False,
        }

        try:
            state = init_state
            for out in app.stream(state):              
                last_state = list(out.values())[-1]
                rover_state = last_state["rover_state"]       
                
                state = last_state
                
                # Safety check to stop episode if it exceeds 100 steps to prevent infinite loops
                if state["step"] > 100:
                    print(f"Episode stopped: exceeded 100 steps")
                    break
            
            if rover_state["distance_meters"] <= ARRIVE_M:
                print("Arrived at target!")
            
            executor.brake(brake_strength=100.0, duration_s=2.0)  # full brake at the end
            plotter.save_plot(f"{step}.png")                   
        finally:        
            logger.close()
            krpc_context.close()

if __name__ == "__main__":
    main()