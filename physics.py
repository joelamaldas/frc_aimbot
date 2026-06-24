from models import GamePiece, Environment, Robot, ShotState
import math

class PhysicsEngine:
    def __init__(self, env: Environment, robot: Robot):
        self.env = env
        self.robot = robot
        
        self.gravity = env.gravity
        self.density = env.air_density

    def calculate_acceleration(self, piece: GamePiece, vx: float, vy: float, vz: float, omega: float, pitch: float, roll: float):
        v = math.sqrt(vx**2 + vy**2 + vz**2)

        # tilted gravity
        gx = self.gravity * math.sin(pitch)
        gy = -self.gravity * math.sin(roll)
        gz = -self.gravity * math.cos(pitch) * math.cos(roll)

        if v == 0:
            return gx, gy, gz

        r_drag = (self.density * piece.drag_coeff * piece.area) / (2.0 * piece.mass)
        s_mag = (self.density * piece.magnus_coeff * piece.radius * piece.area) / (2.0 * piece.mass)

        ax = -r_drag * v * vx + s_mag * omega * vz + gx
        ay = -r_drag * v * vy + gy
        az = gz - r_drag * v * vz - s_mag * omega * vx

        return ax, ay, az
    
    # euler integration tests
    def simulate_euler(self, piece:GamePiece, x:float, y:float, z:float, vx:float, vy:float, vz:float, omega:float, time_step:float):
        while z > 0:
            ax, ay, az = self.calculate_acceleration(piece, vx, vy, vz, omega)
            vx += ax * time_step
            vy += ay * time_step
            vz += az * time_step
            x += vx * time_step
            y += vy * time_step
            z += vz * time_step
        return x, y
    
    # rk4 physics test
    def simulate_rk4(self, piece:GamePiece, x:float, y:float, z:float, vx:float, vy:float, vz:float, omega:float, pitch:float, roll:float, time_step:float, target_z:float, return_path:bool=False):
        h = time_step
        
        path_x, path_y, path_z = [], [], []

        while z > 0:
            if return_path:
                path_x.append(x)
                path_y.append(y)
                path_z.append(z)

            last_x = x
            last_y = y
            last_z = z

            k1_ax, k1_ay, k1_az = self.calculate_acceleration(piece, vx, vy, vz, omega, pitch, roll)
            
            vx2 = vx + h * k1_ax / 2
            vy2 = vy + h * k1_ay / 2
            vz2 = vz + h * k1_az / 2
            k2_ax, k2_ay, k2_az = self.calculate_acceleration(piece, vx2, vy2, vz2, omega, pitch, roll)
            
            vx3 = vx + h * k2_ax / 2
            vy3 = vy + h * k2_ay / 2
            vz3 = vz + h * k2_az / 2
            k3_ax, k3_ay, k3_az = self.calculate_acceleration(piece, vx3, vy3, vz3, omega, pitch, roll)
            
            vx4 = vx + h * k3_ax
            vy4 = vy + h * k3_ay
            vz4 = vz + h * k3_az
            k4_ax, k4_ay, k4_az = self.calculate_acceleration(piece, vx4, vy4, vz4, omega, pitch, roll)
            
            x += (h / 6) * (vx + 2*vx2 + 2*vx3 + vx4)
            y += (h / 6) * (vy + 2*vy2 + 2*vy3 + vy4)
            z += (h / 6) * (vz + 2*vz2 + 2*vz3 + vz4)
            
            vx += (h / 6) * (k1_ax + 2*k2_ax + 2*k3_ax + k4_ax)
            vy += (h / 6) * (k1_ay + 2*k2_ay + 2*k3_ay + k4_ay)
            vz += (h / 6) * (k1_az + 2*k2_az + 2*k3_az + k4_az)

            if vz < 0 and z<= target_z and last_z > target_z:
                dz = last_z - z
                if dz > 1e-6:
                    frac = (last_z - target_z) / dz
                    lx = last_x + frac * (x - last_x)
                    ly = last_y + frac * (y - last_y)
                    if return_path:
                        path_x.append(lx)
                        path_y.append(ly)
                        path_z.append(target_z)
                        return lx, ly, path_x, path_y, path_z
                    return lx, ly
                else:
                    if return_path:
                        path_x.append(x)
                        path_y.append(y)
                        path_z.append(z)
                        return x, y, path_x, path_y, path_z
                    return x, y
            
        if return_path:
            return x, y, path_x, path_y, path_z
        return x, y

    def simulate_shot(self, piece: GamePiece, state: ShotState, rpm: float, hood_deg: float, aim_offset_rad: float, target_z:float, return_path:bool=False):
        lat = self.robot.system_latency
        real_v_rad = state.v_rad + (state.a_rad * lat)
        real_v_tan = state.v_tan + (state.a_tan * lat)
        real_omega = state.omega + (state.alpha * lat)
        
        # speed and spin polynomials
        surface_vel = -4.18e-07 * rpm**2 + 4.75e-03 * rpm - 5.18
        spin = 7.27e-08 * rpm**2 - 6.12e-03 * rpm - 3.64
        
        if surface_vel <= 0:
            return None, None
        
        phi = math.radians(self.robot.hood_offset_deg + hood_deg)

        tip_vx = real_v_rad
        tip_vy = real_v_tan + (real_omega * self.robot.turret_x_offset)
        
        vx = surface_vel * math.cos(phi) * math.cos(aim_offset_rad) + tip_vx
        vy = surface_vel * math.cos(phi) * math.sin(aim_offset_rad) + tip_vy
        vz = surface_vel * math.sin(phi)
        
        return self.simulate_rk4(piece, self.robot.turret_x_offset, self.robot.turret_y_offset, self.robot.turret_z_offset, vx, vy, vz, spin, state.pitch, state.roll, 0.01, target_z, return_path=return_path)