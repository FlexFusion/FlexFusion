import dataclasses
import math

@dataclasses.dataclass
class ExpConfig:
	pp0: int
	pp1: int
	m0: int
	m1: int
	f0_per: int
	f1_per: int
	mem0_per: float
	mem1_per: float

exp_configs = [
	ExpConfig(8, 4, 32, 16, 5, 4, 1.95, 2),
	ExpConfig(8, 4, 16, 8, 5, 4, 1.95, 2),
	ExpConfig(8, 4, 8, 4, 5, 4, 1.95, 2),
	ExpConfig(8, 8, 32, 32, 5, 2, 1.95, 1),
	ExpConfig(8, 8, 16, 16, 5, 2, 1.95, 1),
	ExpConfig(8, 8, 8, 8, 5, 2, 1.95, 1),

	ExpConfig(16, 8, 64, 32, 2, 2, 1.64, 2),
	ExpConfig(16, 8, 32, 16, 2, 2, 1.64, 2),
	ExpConfig(16, 8, 16, 8, 2, 2, 1.64, 2),
	ExpConfig(16, 16, 64, 64, 2, 1, 1.64, 1),
	ExpConfig(16, 16, 32, 32, 2, 1, 1.64, 1),
	ExpConfig(16, 16, 16, 16, 2, 1, 1.64, 1),
]

bounds = []
for c in exp_configs:
	num_nodes = c.pp0

	# The critical path on the large model
	bound0 = (c.m0+c.pp0-1) * (c.f0_per*3)

	# Assume no pipeline bubble
	bound1 = math.ceil((c.pp0*c.m0*(3*c.f0_per) + c.pp1*c.m1*(3*c.f1_per)*(c.pp0/c.pp1)) / num_nodes)

	# All works on a node (assume no bubble)
	bound2 = c.m0*(3*c.f0_per) + c.m1*(3*c.f1_per)

	def get_earlist_fwd_start_time(node_id: int):
		if c.pp0 == c.pp1:
			return min(node_id*c.f0_per, (num_nodes-node_id-1)*c.f1_per)
		else:
			if node_id < c.pp1:
				return min(node_id*c.f0_per, (c.pp1-node_id-1)*c.f1_per)
			else:
				return min(node_id*c.f0_per, (num_nodes-node_id-1)*c.f1_per)
	
	def get_shortest_bwd_tailing_time(node_id: int):
		if c.pp0 == c.pp1:
			return min(node_id*(2*c.f0_per), (num_nodes-node_id-1)*(2*c.f1_per))
		else:
			if node_id < c.pp1:
				return min(node_id*(2*c.f0_per), (c.pp1-node_id-1)*(2*c.f1_per))
			else:
				return min(node_id*(2*c.f0_per), (num_nodes-node_id-1)*(2*c.f1_per))

	# All works on a node (consider the time before the first job arrives & the last job ends)
	bound3 = 0
	for node_id in range(c.pp0):
		cur_node_bound = get_earlist_fwd_start_time(node_id) + bound2 + get_shortest_bwd_tailing_time(node_id)
		bound3 = max(bound3, cur_node_bound)

	bound = max(bound0, bound1, bound2, bound3)
	bounds.append(bound)

print("Bounds on latency:")
print('\n'.join([str(b) for b in bounds]))