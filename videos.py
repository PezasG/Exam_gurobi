import gurobipy as gp
from gurobipy import GRB
import sys

class VideoStreamingOptimizer:
    """
    Optimiseur de placement de vidéos dans des caches.
    """

    def __init__(self, env):
        self.env = env
        self.model = None

        # Données
        self.V = 0
        self.E = 0
        self.R = 0
        self.C = 0
        self.Q = 0

        self.video_sizes = []
        self.dc_latency = []
        self.endpoint_caches = []
        self.cache_latency = []
        self.requests = []

        # Variables
        self.x = {}
        self.y = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.model:
            self.model.dispose()
        return False

    def load_data(self, filename):
        with open(filename) as f:
            lines = [l.strip() for l in f]

        self.V, self.E, self.R, self.C, self.Q = map(int, lines[0].split())
        self.video_sizes = list(map(int, lines[1].split()))

        self.dc_latency = [0]*self.E
        self.endpoint_caches = [[] for _ in range(self.E)]
        self.cache_latency = [{} for _ in range(self.E)]

        idx = 2
        for e in range(self.E):
            dc, k = map(int, lines[idx].split())
            self.dc_latency[e] = dc
            idx += 1
            for _ in range(k):
                c, lat = map(int, lines[idx].split())
                self.endpoint_caches[e].append(c)
                self.cache_latency[e][c] = lat
                idx += 1

        self.requests = [tuple(map(int, lines[idx+i].split())) for i in range(self.R)]

    def build_model(self):
        self.model = gp.Model("VideoStreaming", env=self.env)
        self._create_variables()
        self._add_constraints()
        self._set_objective()

    def _create_variables(self):
        for v in range(self.V):
            for c in range(self.C):
                self.x[v, c] = self.model.addVar(vtype=GRB.BINARY, name=f"x_{v}_{c}")

        for r, (v, e, _) in enumerate(self.requests):
            for c in self.endpoint_caches[e]:
                self.y[r, c] = self.model.addVar(vtype=GRB.BINARY, name=f"y_{r}_{c}")
            self.y[r, self.C] = self.model.addVar(vtype=GRB.BINARY, name=f"y_{r}_dc")

    def _add_constraints(self):
        # Capacité
        for c in range(self.C):
            self.model.addConstr(
                gp.quicksum(self.video_sizes[v]*self.x[v, c] for v in range(self.V)) <= self.Q,
                name=f"cap_{c}"
            )

        # Chaque requête servie une seule fois
        for r, (v, e, _) in enumerate(self.requests):
            sources = [self.y[r, c] for c in self.endpoint_caches[e]] + [self.y[r, self.C]]
            self.model.addConstr(gp.quicksum(sources) == 1, name=f"req_{r}")

        # Cohérence
        for r, (v, e, _) in enumerate(self.requests):
            for c in self.endpoint_caches[e]:
                self.model.addConstr(self.y[r, c] <= self.x[v, c], name=f"link_{r}_{c}")

    def _set_objective(self):
        total_req = sum(n for _, _, n in self.requests)
        K = 1000.0 / total_req
        obj = gp.LinExpr()

        for r, (v, e, n) in enumerate(self.requests):
            for c in self.endpoint_caches[e]:
                obj += K*n*self.cache_latency[e][c]*self.y[r, c]
            obj += K*n*self.dc_latency[e]*self.y[r, self.C]

        self.model.setObjective(obj, GRB.MINIMIZE)

    def optimize(self):
        # On fixe explicitement le MIPGap à 0.5% pour respecter l'énoncé
        self.model.setParam("MIPGap", 0.005)
        self.model.optimize()
        return self.model.Status

    def get_solution(self):
        if self.model.Status != GRB.OPTIMAL:
            return None
        sol = {}
        for c in range(self.C):
            vids = [v for v in range(self.V) if self.x[v, c].X > 0.5]
            if vids:
                sol[c] = vids
        return sol

    def write_solution(self, filename="videos.out"):
        sol = self.get_solution()
        if not sol:
            print("Pas de solution optimale.")
            return
        with open(filename, "w") as f:
            f.write(f"{len(sol)}\n")
            for c in sorted(sol):
                f.write(f"{c} " + " ".join(map(str, sol[c])) + "\n")
        print(f"Solution écrite dans {filename}")

    def export_model(self, filename="videos.mps"):
        self.model.write(filename)
        print(f"Modèle MPS exporté dans {filename}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python videos.py <fichier_input>")
        return
    input_file = sys.argv[1]

    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 1)
        env.setParam("TimeLimit", 300)
        env.start()

        with VideoStreamingOptimizer(env) as opt:
            opt.load_data(input_file)
            opt.build_model()
            opt.export_model("videos.mps")
            status = opt.optimize()
            if status == GRB.OPTIMAL:
                opt.write_solution("videos.out")

if __name__ == "__main__":
    main()
