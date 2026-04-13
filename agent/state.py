import math
import random

class StateTracker:
    def __init__(self):
        self.prev_t = None
        self.prev_dist = None

    def generate_random_target_latlon(self, vessel, sc, max_distance_m=100):
        body = vessel.orbit.body

        theta = random.uniform(0, 2 * math.pi)
        r = max_distance_m * math.sqrt(random.uniform(0, 1))
        dn = r * math.cos(theta)  # north offset (m)
        de = r * math.sin(theta)  # east  offset (m)

        # Offset relative to the vessel position in the surface frame
        # (up, north, east)
        surf_rf = vessel.surface_reference_frame
        vessel_pos_surf = vessel.position(surf_rf)  
        target_pos_surf = (
            vessel_pos_surf[0],            # up 
            vessel_pos_surf[1] + dn,       # north
            vessel_pos_surf[2] + de,       # east
        )

    
        target_pos_body = sc.transform_position(target_pos_surf, surf_rf, body.reference_frame)

        lat = body.latitude_at_position(target_pos_body, body.reference_frame)
        lon = body.longitude_at_position(target_pos_body, body.reference_frame)
        return (lat, lon) # (lat, lon) in degrees in body reference frame
 

    def get_state(self, vessel, target_latlon, sc):
        body = vessel.orbit.body
        body_rf = body.reference_frame
        surf_rf = vessel.surface_reference_frame
        vessel_rf = vessel.reference_frame

        flight = vessel.flight(body_rf)

        vessel_lat_deg = float(flight.latitude)
        vessel_lon_deg = float(flight.longitude)

        # Orientation
        heading_deg = float(flight.heading) # 0-360, where 0/360 is north, 90 is east
        pitch_deg = float(flight.pitch) # positive => nose up, negative => nose down
        roll_deg = float(flight.roll) # positive => right side up, negative => left side up (banking)

        tlat, tlon = float(target_latlon[0]), float(target_latlon[1])

        try:
            target_pos_body = body.surface_position(tlat, tlon, body_rf)
        except TypeError:
            target_pos_body = body.surface_position(tlat, tlon, 0.0, body_rf)

        target_pos_surf = sc.transform_position(target_pos_body, body_rf, surf_rf)

        delta_up, delta_north, delta_east = target_pos_surf
        dist_m = math.hypot(delta_north, delta_east)

        # convert to surface reference frame and extract forward/right components
        forward_direction = sc.transform_direction((0.0, 1.0, 0.0), vessel_rf, surf_rf)
        right_direction = sc.transform_direction((1.0, 0.0, 0.0), vessel_rf, surf_rf)

        # surf_rf is (up, north, east)
        fh_up, fh_n, fh_e = forward_direction
        rh_up, rh_n, rh_e = right_direction

        # Normalize to get unit vectors (and handle division by zero just in case)
        fh_len = math.hypot(fh_n, fh_e)
        if fh_len < 1e-6:
            fh_n, fh_e = 1.0, 0.0
        else:
            fh_n, fh_e = fh_n / fh_len, fh_e / fh_len

        # Gram-Schmidt Orthonormalization (make sure forward and right vectors are perpendicular in case the vessel is tilted or numerical issues)
        dot = rh_n * fh_n + rh_e * fh_e
        rh_n, rh_e = rh_n - dot * fh_n, rh_e - dot * fh_e
        rh_len = math.hypot(rh_n, rh_e)
        if rh_len < 1e-6:
            rh_n, rh_e = -fh_e, fh_n
        else:
            rh_n, rh_e = rh_n / rh_len, rh_e / rh_len

        # Project target offset into rover frame (delta_north/east are now correct)
        target_fwd_m   = delta_north * fh_n + delta_east * fh_e
        target_right_m = delta_north * rh_n + delta_east * rh_e

        # Velocity (EAST, NORTH, UP)
        v_body = vessel.velocity(body_rf)
        v_surf = sc.transform_direction(v_body, body_rf, surf_rf)
        vel_up, vel_north, vel_east = v_surf
        speed_h_mps = math.hypot(vel_north, vel_east)

        bearing_err_rad = math.atan2(target_right_m, target_fwd_m)  # + => target to the right
        bearing_err_deg = math.degrees(bearing_err_rad)

        return {
            "forward_distance_meters": float(target_fwd_m),
            "right_distance_meters": float(target_right_m),            
            "distance_meters": float(dist_m),
            "speed_mps": float(speed_h_mps),
            "vessel_latitude_deg": vessel_lat_deg,
            "vessel_longitude_deg": vessel_lon_deg,
            "target_latitude_deg": tlat,
            "target_longitude_deg": tlon,
            "bearing_error_deg": bearing_err_deg,
            "heading_deg": heading_deg,
            "pitch_deg": pitch_deg,
            "roll_deg": roll_deg
        }


