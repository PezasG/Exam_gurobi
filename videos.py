"""
Video Streaming Optimization - Hash Code 2017
MIP Model Implementation using Gurobi
"""

import gurobipy as gp
from gurobipy import GRB


class VideoStreamingOptimizer:
    """Optimizer for video streaming cache allocation problem."""
    
    def __init__(self, env):
        """Initialize optimizer with Gurobi environment.
        
        Args:
            env: Gurobi environment
        """
        self.env = env
        self.model = None
        
        # Data attributes
        self.V = 0  # Number of videos
        self.E = 0  # Number of endpoints
        self.R = 0  # Number of requests
        self.C = 0  # Number of cache servers
        self.Q = 0  # Cache capacity in MB
        
        self.video_sizes = []  # Size of each video in MB
        self.endpoint_dc_latency = []  # Latency from data center to each endpoint
        self.endpoint_caches = []  # List of connected caches for each endpoint
        self.cache_latencies = []  # cache_latencies[e][c] = latency from cache c to endpoint e
        
        self.requests = []  # List of (video_id, endpoint_id, num_requests)
        
        # Decision variables
        self.x = {}  # x[v,c] = 1 if video v is stored in cache c
        self.y = {}  # y[r,c] = 1 if request r is served by cache c
        
    def __enter__(self):
        """Enter context manager."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and dispose model."""
        if self.model is not None:
            self.model.dispose()
        return False
    
    def load_data_from_file(self, filename):
        """Load problem data from input file.
        
        Args:
            filename: Path to input file
        """
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Parse first line
        first_line = lines[0].strip().split()
        self.V = int(first_line[0])
        self.E = int(first_line[1])
        self.R = int(first_line[2])
        self.C = int(first_line[3])
        self.Q = int(first_line[4])
        
        # Parse video sizes
        self.video_sizes = list(map(int, lines[1].strip().split()))
        
        # Initialize endpoint data structures
        self.endpoint_dc_latency = [0] * self.E
        self.endpoint_caches = [[] for _ in range(self.E)]
        self.cache_latencies = [{} for _ in range(self.E)]
        
        # Parse endpoints
        line_idx = 2
        for e in range(self.E):
            endpoint_line = lines[line_idx].strip().split()
            self.endpoint_dc_latency[e] = int(endpoint_line[0])
            K = int(endpoint_line[1])  # Number of connected caches
            line_idx += 1
            
            for _ in range(K):
                cache_line = lines[line_idx].strip().split()
                cache_id = int(cache_line[0])
                latency = int(cache_line[1])
                self.endpoint_caches[e].append(cache_id)
                self.cache_latencies[e][cache_id] = latency
                line_idx += 1
        
        # Parse requests
        self.requests = []
        for _ in range(self.R):
            request_line = lines[line_idx].strip().split()
            video_id = int(request_line[0])
            endpoint_id = int(request_line[1])
            num_requests = int(request_line[2])
            self.requests.append((video_id, endpoint_id, num_requests))
            line_idx += 1
    
    def set_data(self, V, E, R, C, Q, video_sizes, endpoint_dc_latency, 
                 endpoint_caches, cache_latencies, requests):
        """Set problem data manually.
        
        Args:
            V: Number of videos
            E: Number of endpoints
            R: Number of requests
            C: Number of cache servers
            Q: Cache capacity
            video_sizes: List of video sizes
            endpoint_dc_latency: List of latencies from data center to endpoints
            endpoint_caches: List of lists of connected caches per endpoint
            cache_latencies: List of dicts mapping cache_id to latency for each endpoint
            requests: List of (video_id, endpoint_id, num_requests)
        """
        self.V = V
        self.E = E
        self.R = R
        self.C = C
        self.Q = Q
        self.video_sizes = video_sizes
        self.endpoint_dc_latency = endpoint_dc_latency
        self.endpoint_caches = endpoint_caches
        self.cache_latencies = cache_latencies
        self.requests = requests
    
    def build_model(self):
        """Build the optimization model."""
        print("Building optimization model...")
        
        # Create model
        self.model = gp.Model("VideoStreaming", env=self.env)
        
        # Create variables
        self._create_variables()
        
        # Add constraints
        self._add_constraints()
        
        # Set objective
        self._set_objective()
        
        print(f"Model built: {self.model.NumVars} variables, {self.model.NumConstrs} constraints")
    
    def _create_variables(self):
        """Create decision variables."""
        print("Creating variables...")
        
        # x[v,c]: 1 if video v is stored in cache c
        self.x = {}
        for v in range(self.V):
            for c in range(self.C):
                self.x[v, c] = self.model.addVar(
                    vtype=GRB.BINARY,
                    name=f"x_v{v}_c{c}"
                )
        
        # y[r,c]: 1 if request r is served by cache c
        # Note: We include the data center as a "virtual cache" with index C
        self.y = {}
        for r in range(self.R):
            video_id, endpoint_id, _ = self.requests[r]
            
            # Add variable for each connected cache
            for c in self.endpoint_caches[endpoint_id]:
                self.y[r, c] = self.model.addVar(
                    vtype=GRB.BINARY,
                    name=f"y_r{r}_c{c}"
                )
            
            # Add variable for data center (virtual cache with index C)
            self.y[r, self.C] = self.model.addVar(
                vtype=GRB.BINARY,
                name=f"y_r{r}_DC"
            )
    
    def _add_constraints(self):
        """Add constraints to the model."""
        print("Adding constraints...")
        
        # Constraint [1]: Cache capacity constraints
        for c in range(self.C):
            self.model.addConstr(
                gp.quicksum(self.video_sizes[v] * self.x[v, c] for v in range(self.V)) <= self.Q,
                name=f"capacity_c{c}"
            )
        
        # Constraint [2]: Each request must be served by exactly one source
        for r in range(self.R):
            video_id, endpoint_id, _ = self.requests[r]
            
            # Sum over all connected caches plus data center
            sources = [self.y[r, c] for c in self.endpoint_caches[endpoint_id]]
            sources.append(self.y[r, self.C])  # Add data center
            
            self.model.addConstr(
                gp.quicksum(sources) == 1,
                name=f"serve_r{r}"
            )
        
        # Constraint [3]: Can only serve from cache if video is stored there
        for r in range(self.R):
            video_id, endpoint_id, _ = self.requests[r]
            
            for c in self.endpoint_caches[endpoint_id]:
                self.model.addConstr(
                    self.y[r, c] <= self.x[video_id, c],
                    name=f"availability_r{r}_c{c}"
                )
    
    def _set_objective(self):
        """Set the objective function."""
        print("Setting objective...")
        
        # Calculate K = 1000 / sum(n_r)
        total_requests = sum(num_req for _, _, num_req in self.requests)
        K = 1000.0 / total_requests
        
        # Objective: minimize average latency (time saved is maximized)
        # min K * sum_{r,c} n_r * (l_D_e - l_c_e) * y_{r,c}
        # Which is equivalent to: min K * sum_{r,c} n_r * l_c_e * y_{r,c} - constant
        
        objective = gp.LinExpr()
        
        for r in range(self.R):
            video_id, endpoint_id, num_requests = self.requests[r]
            dc_latency = self.endpoint_dc_latency[endpoint_id]
            
            # For each connected cache
            for c in self.endpoint_caches[endpoint_id]:
                cache_latency = self.cache_latencies[endpoint_id][c]
                # Cost is proportional to latency
                cost = K * num_requests * cache_latency
                objective += cost * self.y[r, c]
            
            # For data center (always available with dc_latency)
            cost = K * num_requests * dc_latency
            objective += cost * self.y[r, self.C]
        
        self.model.setObjective(objective, GRB.MINIMIZE)
    
    def optimize(self):
        """Solve the optimization model."""
        print("\nOptimizing...")
        self.model.optimize()
        
        # Check optimization status
        status = self.model.Status
        if status == GRB.OPTIMAL:
            print(f"\nOptimal solution found!")
            print(f"Objective value: {self.model.ObjVal:.2f}")
            return status
        elif status == GRB.INFEASIBLE:
            print("\nModel is infeasible.")
            self.model.computeIIS()
            print("IIS computed. Printing IIS constraints:")
            for c in self.model.getConstrs():
                if c.IISConstr:
                    print(f"  {c.ConstrName}")
            return status
        elif status == GRB.UNBOUNDED:
            print("\nModel is unbounded.")
            return status
        else:
            print(f"\nOptimization ended with status {status}")
            return status
    
    def get_solution(self):
        """Extract the solution.
        
        Returns:
            Dictionary mapping cache_id to list of video_ids stored in that cache
        """
        if self.model.Status != GRB.OPTIMAL:
            print("No optimal solution available.")
            return None
        
        solution = {}
        for c in range(self.C):
            videos_in_cache = []
            for v in range(self.V):
                if self.x[v, c].X > 0.5:  # Binary variable is 1
                    videos_in_cache.append(v)
            if videos_in_cache:  # Only include non-empty caches
                solution[c] = videos_in_cache
        
        return solution
    
    def display_solution(self):
        """Display the solution in a readable format."""
        solution = self.get_solution()
        if solution is None:
            return
        
        print("\n" + "="*60)
        print("SOLUTION")
        print("="*60)
        
        for cache_id in sorted(solution.keys()):
            videos = solution[cache_id]
            total_size = sum(self.video_sizes[v] for v in videos)
            print(f"\nCache {cache_id} (capacity: {self.Q} MB, used: {total_size} MB):")
            print(f"  Videos: {videos}")
        
        # Calculate actual score
        score = self.calculate_score(solution)
        print(f"\n{'='*60}")
        print(f"Total Score: {score:,}")
        print("="*60)
    
    def calculate_score(self, solution):
        """Calculate the score according to the problem definition.
        
        Args:
            solution: Dictionary mapping cache_id to list of video_ids
            
        Returns:
            Score in microseconds
        """
        total_time_saved = 0
        total_requests = 0
        
        for r in range(self.R):
            video_id, endpoint_id, num_requests = self.requests[r]
            dc_latency = self.endpoint_dc_latency[endpoint_id]
            
            # Find the best latency for this request
            best_latency = dc_latency  # Default to data center
            
            for cache_id in self.endpoint_caches[endpoint_id]:
                if cache_id in solution and video_id in solution[cache_id]:
                    cache_latency = self.cache_latencies[endpoint_id][cache_id]
                    best_latency = min(best_latency, cache_latency)
            
            # Time saved per request
            time_saved = dc_latency - best_latency
            total_time_saved += num_requests * time_saved
            total_requests += num_requests
        
        # Convert to microseconds and calculate average
        score = int((total_time_saved * 1000) / total_requests)
        return score
    
    def write_solution(self, filename):
        """Write solution to output file in the required format.
        
        Args:
            filename: Output file path
        """
        solution = self.get_solution()
        if solution is None:
            print("No solution to write.")
            return
        
        with open(filename, 'w') as f:
            # First line: number of cache descriptions
            f.write(f"{len(solution)}\n")
            
            # Each subsequent line: cache_id followed by video_ids
            for cache_id in sorted(solution.keys()):
                videos = solution[cache_id]
                line = f"{cache_id} " + " ".join(map(str, videos))
                f.write(line + "\n")
        
        print(f"\nSolution written to {filename}")


def main():
    """Main function demonstrating the optimizer usage."""
    
    # Example data from the problem statement
    V = 5  # 5 videos
    E = 2  # 2 endpoints
    R = 4  # 4 requests
    C = 3  # 3 cache servers
    Q = 100  # 100 MB capacity
    
    video_sizes = [50, 50, 80, 30, 110]
    
    endpoint_dc_latency = [1000, 500]
    
    # Endpoint 0 is connected to caches 0, 2, 1
    # Endpoint 1 is not connected to any cache
    endpoint_caches = [
        [0, 2, 1],  # Endpoint 0
        []          # Endpoint 1
    ]
    
    cache_latencies = [
        {0: 100, 2: 200, 1: 300},  # Endpoint 0
        {}                          # Endpoint 1
    ]
    
    requests = [
        (3, 0, 1500),  # 1500 requests for video 3 from endpoint 0
        (0, 1, 1000),  # 1000 requests for video 0 from endpoint 1
        (4, 0, 500),   # 500 requests for video 4 from endpoint 0
        (1, 0, 1000),  # 1000 requests for video 1 from endpoint 0
    ]
    
    # Create Gurobi environment
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 1)
        env.setParam('TimeLimit', 300)  # 5 minutes
        env.start()
        
        # Create and use optimizer
        with VideoStreamingOptimizer(env) as optimizer:
            # Set data
            optimizer.set_data(V, E, R, C, Q, video_sizes, endpoint_dc_latency,
                             endpoint_caches, cache_latencies, requests)
            
            # Build model
            optimizer.build_model()
            
            # Optimize
            status = optimizer.optimize()
            
            # Display results
            if status == GRB.OPTIMAL:
                optimizer.display_solution()
                # optimizer.write_solution("output.txt")


if __name__ == "__main__":
    main()
