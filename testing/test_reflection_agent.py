from application.agents.reflection_agent import ReflectionAgent


reflection_agent = ReflectionAgent()


question = """
Explain Azure Stack HCI upgrade process
"""


answer = """
The Azure Stack HCI upgrade process requires running
environment validation checks, preparing the cluster,
checking compatibility, performing upgrade operations,
and verifying the cluster health after completion.
"""


context = [

    {

        "source": "Upgrade Guide",

        "text": """
        Azure Stack HCI upgrade requires environment
        checker validation, readiness checks,
        cluster upgrade execution and post upgrade
        verification.
        """

    }

]


result = reflection_agent.evaluate(

    question,

    answer,

    context

)


print("\nReflection Result")

print("=" * 50)


for key, value in result.items():

    print(
        key,
        ":",
        value
    )