import math
import random
import numpy as np
from scipy.optimize import minimize as sp_minimize
import csv
from tqdm import tqdm
from config import engine, hub, fuel_2026
from physics import PhysicsEngine                                                                                                                                                                                                                                         
from models import Target, GamePiece, ShotState  

class DatasetGenerator:
    def __init__(self, engine: PhysicsEngine, targets: list[Target], piece: GamePiece):
        self.engine = engine
        self.targets = targets
        self.piece = piece

        self.static_lookups = {}
        self.static_distances = np.linspace(1.0, 20.0, 200)

    def precompute_static_shots(self):
        for target in self.targets:
            print(f"computing {target.name} static shots...")
            print(f"optimizing {target.name} shots...")
            rpms = []
            hoods = []
            prev_rpm, prev_hood = 3200, 9.5
            for target_d in self.static_distances:
                def cost(x):
                    state = ShotState(
                    v_rad = 0.0,
                    v_tan = 0.0,
                    omega = 0.0,
                    a_rad = 0.0,
                    a_tan = 0.0,
                    alpha = 0.0,
                    pitch = 0.0,
                    roll = 0.0,
                    distance = 0.0
                    )
                    lx, ly = self.engine.simulate_shot(self.piece, state, rpm=x[0], hood_deg=x[1], aim_offset_rad=0.0, target_z=target.height)
                    
                    if lx is None:
                        return 1000000.0
                    
                    dist = math.sqrt(lx**2 + ly**2)
                    miss = (dist - target_d) ** 2
                    smooth = 1e-6 * ((x[0] - prev_rpm)**2 + 1e4 * (x[1] - prev_hood)**2)
                    low_rpm_penalty = 1e-7 * x[0]**2
                    return miss + smooth + low_rpm_penalty
                
                res = sp_minimize(cost, [prev_rpm, prev_hood], method='Nelder-Mead')
                winning_rpm = res.x[0]
                winning_hood = res.x[1]

                prev_rpm = winning_rpm
                prev_hood = winning_hood

                rpms.append(winning_rpm)
                hoods.append(winning_hood)

            self.static_lookups[target.name] = {"rpms": rpms, "hoods": hoods}    
            pass

    def generate_reverse_mapping_dataset(self, num_shots: int):

        # print(f"generating {num_shots} shots...")

        dataset = []

        for _ in tqdm(range(num_shots), desc="generating dataset"):
            target = random.choice(self.targets)

            base_dist = random.uniform(1.0, 20.0)
            
            target_rpms = self.static_lookups[target.name]["rpms"]
            target_hoods = self.static_lookups[target.name]["hoods"]

            base_rpm = np.interp(base_dist, self.static_distances, target_rpms)
            base_hood = np.interp(base_dist, self.static_distances, target_hoods)

            rpm = base_rpm + random.uniform(-15.0, 15.0)
            hood = base_hood + random.uniform(-0.5, 0.5)

            state = ShotState(
                v_rad = random.uniform(-5.0, 5.0),
                v_tan = random.uniform(-5.0, 5.0),
                omega = random.uniform(-4.0, 4.0),
                a_rad = random.uniform(-8.0, 8.0),
                a_tan = random.uniform(-8.0, 8.0),
                alpha = random.uniform(-10.0, 10.0),
                pitch = random.uniform(-0.3, 0.3),
                roll  = random.uniform(-0.3, 0.3),
                distance = 0.0 
            )

            aim_offset = math.radians(random.uniform(-20, 20))

            lx, ly = self.engine.simulate_shot(
                piece = self.piece,
                state = state,
                rpm = rpm,
                hood_deg = hood,
                aim_offset_rad = aim_offset,
                target_z = target.height
            )

            if lx is None:
                continue

            landing_dist = math.sqrt(lx**2 + ly**2)

            naive_angle = math.atan2(ly, lx)

            turret_corr = aim_offset - naive_angle

            turret_corr = (turret_corr + math.pi) % (2 * math.pi) - math.pi

            cos_a = lx / landing_dist                                                                                                                                                                                                                                    
            sin_a = ly / landing_dist                                                                                                                                                                                                                                    
                                                                                                                                                                                                                                                                             
            v_rad_eff = state.v_rad * cos_a + state.v_tan * sin_a                                                                                                                                                                                                        
            v_tan_eff = -state.v_rad * sin_a + state.v_tan * cos_a                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                             
            a_rad_eff = state.a_rad * cos_a + state.a_tan * sin_a                                                                                                                                                                                                        
            a_tan_eff = -state.a_rad * sin_a + state.a_tan * cos_a

            inputs = [landing_dist, target.height, v_rad_eff, v_tan_eff, state.omega, a_rad_eff, a_tan_eff, state.alpha, state.pitch, state.roll]

            labels = [rpm, hood, turret_corr]

        
            # throw away bad shots from dataset

            if landing_dist < 0.5 or landing_dist > 16.0:
                continue

            if abs(turret_corr) > math.radians(35):
                continue


            dataset.append(inputs + labels)

        return dataset

if __name__ == "__main__":
    generator = DatasetGenerator(engine, hub, fuel_2026)
    generator.precompute_static_shots()
    dataset = generator.generate_reverse_mapping_dataset(500000)
    # print(dataset[0])
    with open("output/training_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(dataset)

