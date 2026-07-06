from application.agents.planner_agent import PlannerAgent


planner = PlannerAgent()


questions = [

    "How to setup environment checker?",

    "Which command installs environment checker?",

    "Summarize Azure Stack HCI upgrade process"

]


for question in questions:

    print("=" * 50)

    print("Question:")

    print(question)

    print()

    result = planner.analyze_query(question)

    print(result)