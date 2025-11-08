from pyomo.environ import *

# Create model
model = ConcreteModel()

#######################################################################
###                            PARAMETERS                           ###
#######################################################################
model.P = Set(initialize=['Shirts', 'Shorts', 'Pants', 'Skirts', 'Jackets'])

# Parameters
model.variable_cost = Param(model.P, initialize={
    'Shirts': 20, 'Shorts': 10, 'Pants': 25, 'Skirts': 30, 'Jackets': 35
})

model.selling_price = Param(model.P, initialize={
    'Shirts': 35, 'Shorts': 40, 'Pants': 65, 'Skirts': 70, 'Jackets': 110
})

model.fixed_cost = Param(model.P, initialize={
    'Shirts': 1500, 'Shorts': 1200, 'Pants': 1600, 'Skirts': 1500, 'Jackets': 1600
})

model.labor_hours = Param(model.P, initialize={
    'Shirts': 2.0, 'Shorts': 1.0, 'Pants': 6.0, 'Skirts': 4.0, 'Jackets': 8.0
})

model.cloth = Param(model.P, initialize={
    'Shirts': 3.0, 'Shorts': 2.5, 'Pants': 4.0, 'Skirts': 4.5, 'Jackets': 5.5
})

model.effective_capacity = Param(model.P, initialize={
    'Shirts': 1500, 'Shorts': 2250, 'Pants': 666.6667, 'Skirts': 1000, 'Jackets': 500
})

# Available resources
model.available_labor = 4000
model.available_cloth = 4500

#######################################################################
###                 DECISION VARS & OBJECTIVE FUNC                  ###
#######################################################################
model.x = Var(model.P, domain=NonNegativeIntegers)  # Units produced
model.y = Var(model.P, domain=Binary)  # Rent equipment?

# Objective: maximize profit
def profit_rule(model):
    revenue = sum(model.selling_price[p] * model.x[p] for p in model.P)
    variable_cost = sum(model.variable_cost[p] * model.x[p] for p in model.P)
    fixed_cost = sum(model.fixed_cost[p] * model.y[p] for p in model.P)
    return revenue - variable_cost - fixed_cost

model.profit = Objective(rule=profit_rule, sense=maximize)



#######################################################################
###                          CONSTRAINTS                            ###
#######################################################################

# Constraint 1: Effective capacity
def capacity_rule(model, p):
    return model.x[p] <= model.effective_capacity[p] * model.y[p]
model.capacity_constraint = Constraint(model.P, rule=capacity_rule)

# Constraint 2: Labor hours
def labor_rule(model):
    return sum(model.labor_hours[p] * model.x[p] for p in model.P) <= model.available_labor
model.labor_constraint = Constraint(rule=labor_rule)

# Constraint 3: Cloth availability
def cloth_rule(model):
    return sum(model.cloth[p] * model.x[p] for p in model.P) <= model.available_cloth
model.cloth_constraint = Constraint(rule=cloth_rule)




#######################################################################
###                         SOLVING & RESULTS                       ###
#######################################################################
solver = SolverFactory('glpk')
results = solver.solve(model)

# Results
print("=== OPTIMAL SOLUTION ===")
print(f"Total profit: {model.profit():.2f} €")

print("\nEquipment rental decisions:")
for p in model.P:
    print(f"  {p}: {'YES' if model.y[p]() > 0.5 else 'NO'}")

print("\nUnits to produce:")
for p in model.P:
    if model.x[p]() > 0:
        print(f"  {p}: {model.x[p]():.0f} units")

print("\nResource utilization:")
labor_used = sum(model.labor_hours[p] * model.x[p]() for p in model.P)
cloth_used = sum(model.cloth[p] * model.x[p]() for p in model.P)
print(f"  Labor hours used: {labor_used:.1f} / {model.available_labor}")
print(f"  Cloth used: {cloth_used:.1f} / {model.available_cloth}")

# Economic analysis
print("\nEconomic analysis:")
revenue = sum(model.selling_price[p] * model.x[p]() for p in model.P)
variable_cost_total = sum(model.variable_cost[p] * model.x[p]() for p in model.P)
fixed_cost_total = sum(model.fixed_cost[p] * model.y[p]() for p in model.P)
print(f"  Revenue: {revenue:.2f} €")
print(f"  Variable costs: {variable_cost_total:.2f} €")
print(f"  Fixed costs: {fixed_cost_total:.2f} €")
print(f"  Profit: {revenue - variable_cost_total - fixed_cost_total:.2f} €")
