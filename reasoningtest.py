import ollama
import json
import time

def solve_problem(initial_query):
    # Initial breakdown prompt
    breakdown_prompt = f"""Break this problem into 3-5 sequential steps. Format each step as:
Step N: description

***WRITE HIGH LEVEL STEPS ONLY, DO NOT DO ANY CALCULATIONS***

Separate each step with two newlines.

Problem: {initial_query}"""

    print("🧠 Initial Problem Breakdown")
    try:
        breakdown_response = ollama.chat(
            model='MFDoom/deepseek-r1-tool-calling:14b',
            messages=[{'role': 'user', 'content': breakdown_prompt}],
            options={'temperature': 0.1}
        )
        
        # Extract answer after think tags and parse steps
        response_content = breakdown_response['message']['content']
        if '</think>' in response_content:
            answer = response_content.split('</think>')[-1].strip()
            # Split by double newlines and filter valid steps
            steps = [step.strip() for step in answer.split('\n\n') if step.strip() and step.lower().startswith('step')]
        else:
            steps = []
        
        if not steps or len(steps) > 5:
            raise ValueError("Invalid number of steps in response")
            
        print(f"📋 Breakdown into {len(steps)} steps:")
        for step in steps:
            print(f"  {step}")
    except Exception as e:
        print(f"❌ Breakdown failed: {str(e)}")
        return

    # Sequential solving with context
    solutions = []
    for step_num, current_step in enumerate(steps, 1):
        # Build context from previous solutions
        context = "\n\n".join([
            f"{steps[i]}\nSolution: {sol}" 
            for i, sol in enumerate(solutions)
        ])

        step_prompt = f"""Solve this step of the problem.

Problem: {initial_query}
Current Step: {current_step}

Previous Solutions:
{context if context else "No previous steps completed yet"}

Provide a detailed solution for this specific step."""

        print(f"\n🔧 Processing Step {step_num}/{len(steps)}")
        
        try:
            start_time = time.time()
            step_response = ollama.chat(
                model='MFDoom/deepseek-r1-tool-calling:14b',
                messages=[{'role': 'user', 'content': step_prompt}],
                options={'temperature': 0.3}
            )
            elapsed = time.time() - start_time
            
            response_content = step_response['message']['content']
            if '</think>' in response_content:
                solution = response_content.split('</think>')[-1].strip()
            else:
                solution = "No solution found in response"
            
            print(f"✅ Step {step_num} solved in {elapsed:.2f}s")
            print(f"   Solution Preview: {solution[:120]}...")
            solutions.append(solution)
            
        except Exception as e:
            print(f"❌ Step {step_num} failed: {str(e)}")
            solutions.append("Step could not be completed")

    # Final output with complete solutions
    print("\n📚 Final Solution Breakdown:")
    for i, (step, solution) in enumerate(zip(steps, solutions), 1):
        print(f"\n{step}")
        print(f"Solution: {solution}")
        print("-" * 50)

# Test with a problem
solve_problem("10 moles of an ideal gas expands reversibly and isothermally from a pressure of 10 atm to 2 atm at 300 K. Calculate the work done.")
