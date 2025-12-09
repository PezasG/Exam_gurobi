import gurobipy as gp
from gurobipy import GRB
import sys

class VideoStreamingOptimizer:
    """Optimiseur de placement de vidéos dans des caches pour gros datasets avec traces."""

    def __init__(self, env):
        self.env = env
        self.model = None

        # Données
        self.V = self.E = self.R = self.C = self.Q = 0
        self.video_sizes = []
        self.dc_latency = []
        self.endpoint_caches = []
        self.cache_latency = []
        self.requests = []

        # Variables de décision
        self.x = {}
        self.y = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.model:
            self.model.dispose()
        return False

    def load_data(self, filename):
        print(f"[INFO] Lecture des données depuis '{filename}' ...")
        with open(filename, encoding="utf-8") as f:
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
        print(f"[INFO] {self.V} vidéos, {self.E} endpoints, {self.R} requêtes, {self.C} caches, capacité {self.Q} MB")

    def build_model(self):
        print("[INFO] Construction du modèle Gurobi ...")
        self.model = gp.Model("VideoStreaming", env=self.env)

        # Paramètres pour gros datasets
        self.model.setParam("MIPGap", 0.005)   # 0.5% gap
        self.model.setParam("Threads", 12)     # tous les coeurs disponibles
        self.model.setParam("Presolve", 2)
        self.model.setParam("Heuristics", 0.5)
        self.model.setParam("OutputFlag", 1)

        self._create_variables()
        self._add_constraints()
        self._set_objective()
        print("[INFO] Modèle construit.")

    def _create_variables(self):
        print("[INFO] Création des variables ...")
        for v in range(self.V):
            for c in range(self.C):
                if self.video_sizes[v] <= self.Q:  # ignore vidéos trop grosses
                    self.x[v, c] = self.model.addVar(vtype=GRB.BINARY, name=f"x_{v}_{c}")

        for r, (v, e, _) in enumerate(self.requests):
            for c in self.endpoint_caches[e]:
                if (v, c) in self.x:
                    self.y[r, c] = self.model.addVar(vtype=GRB.BINARY, name=f"y_{r}_{c}")
            self.y[r, self.C] = self.model.addVar(vtype=GRB.BINARY, name=f"y_{r}_dc")

    def _add_constraints(self):
        print("[INFO] Ajout des contraintes ...")
        for c in range(self.C):
            vids_in_cache = [v for v in range(self.V) if (v, c) in self.x]
            if vids_in_cache:
                self.model.addConstr(
                    gp.quicksum(self.video_sizes[v]*self.x[v, c] for v in vids_in_cache) <= self.Q,
                    name=f"cap_{c}"
                )

        for r, (v, e, _) in enumerate(self.requests):
            sources = [self.y[r, c] for c in self.endpoint_caches[e] if (r, c) in self.y]
            sources.append(self.y[r, self.C])
            self.model.addConstr(gp.quicksum(sources) == 1, name=f"req_{r}")

        for r, (v, e, _) in enumerate(self.requests):
            for c in self.endpoint_caches[e]:
                if (r, c) in self.y and (v, c) in self.x:
                    self.model.addConstr(self.y[r, c] <= self.x[v, c], name=f"link_{r}_{c}")

    def _set_objective(self):
        print("[INFO] Définition de l'objectif ...")
        total_req = sum(n for _, _, n in self.requests)
        K = 1000.0 / total_req
        obj = gp.LinExpr()
        for r, (v, e, n) in enumerate(self.requests):
            for c in self.endpoint_caches[e]:
                if (r, c) in self.y:
                    obj += K * n * self.cache_latency[e][c] * self.y[r, c]
            obj += K * n * self.dc_latency[e] * self.y[r, self.C]
        self.model.setObjective(obj, GRB.MINIMIZE)

    def optimize(self):
        print("[INFO] Optimisation en cours ...")
        self.model.optimize()
        print(f"[INFO] Statut optimisation: {self.model.Status}")
        return self.model.Status

    def get_solution(self):
        if self.model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
            return None
        sol = {}
        for c in range(self.C):
            vids = [v for v in range(self.V) if (v, c) in self.x and self.x[v, c].X > 0.5]
            if vids:
                sol[c] = vids
        return sol

    def write_solution(self, filename="videos.out"):
        sol = self.get_solution()
        if not sol:
            print("[WARN] Pas de solution optimale trouvée.")
            return
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"{len(sol)}\n")
            for c in sorted(sol):
                f.write(f"{c} " + " ".join(map(str, sol[c])) + "\n")
        print(f"[INFO] Solution écrite dans '{filename}'")

    def export_model(self, filename="videos.mps"):
        self.model.write(filename)
        print(f"[INFO] Modèle MPS exporté dans '{filename}'")

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
            if status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
                opt.write_solution("videos.out")

if __name__ == "__main__":
    main()
