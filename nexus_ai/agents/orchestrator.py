import json

class OrchestratorAgent:
    def __init__(self, planner_agent, agents_dict):
        self.planner = planner_agent
        self.agents = agents_dict
    
    async def execute(self, user_goal: str) -> str:
        print(f"Goal: {user_goal}")
        print(f"{'='*70}\n")
    
        print("Phase 1: Planning...")
        plan_response = await self.planner.run(user_goal)
        
        plan = self._parse_plan(plan_response)
        steps = plan.get("steps", [])
        print(f"Plan created with {len(steps)} steps\n")
        
        for i, step in enumerate(steps, 1):
            print(f"   {i}. {step['agent']}: {step['task']}")
        print()
        
        print("Phase 2: Execution...")
        results = []
        
        for i, step in enumerate(steps, 1):
            agent_name = step.get("agent")
            task = step.get("task")
            
            print(f"\n[{i}/{len(steps)}] {agent_name}: {task}")
            
            if agent_name not in self.agents:
                print(f"Agent '{agent_name}' not found, skipping")
                continue
            
            context = self._build_context(task, results)
            
            agent = self.agents[agent_name]
            result = await agent.run(context)
            
            results.append({
                "agent": agent_name,
                "task": task,
                "output": result
            })
        
        print(f"\n{'='*70}")
        print("Execution Complete!")
        print(f"{'='*70}\n")
        
        return results[-1]['output']
    
    def _parse_plan(self, plan_response: str) -> dict:
        try:
            return json.loads(plan_response)
        except json.JSONDecodeError:
            start = plan_response.find("{")
            end = plan_response.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(plan_response[start:end])
            raise ValueError(f"Could not parse plan: {plan_response}")
    
    def _build_context(self, task: str, previous_results: list) -> str:
        context = f"Task: {task}\n\n"
        
        if previous_results:
            context += "Context from previous steps:\n"
            for r in previous_results[-1:]:
                truncated = r['output'][:500] + "..." if len(r['output']) > 500 else r['output']
                context += f"\n{r['agent']}: {truncated}\n"
        
        return context
