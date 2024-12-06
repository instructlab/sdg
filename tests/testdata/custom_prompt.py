# SPDX-License-Identifier: Apache-2.0

# First Party
from instructlab.sdg import PromptRegistry


# Register our custom chat template under the "custom_model_family"
# model family
@PromptRegistry.register("custom_model_family")
def custom_chat_template():
    return """{% for message in messages %}{% if message['role'] == 'system' %}{{ '<<SYSTEM>>' + '\n' + message['content'] + '\n' }}{% elif message['role'] == 'user' %}{{ '<<USER>>' + '\n' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{{ '<<ASSISTANT>>' + '\n' + message['content'] + ('' if loop.last else '\n') }}{% endif %}{% endfor %}"""


# Lookup the chat template for "custom_model_family" model family
template = PromptRegistry.get_template("custom_model_family")
assert template is not None

# Ensure the template found is our custom one
prompt = template.render(
    messages=[
        {"role": "system", "content": "system prompt goes here"},
        {"role": "user", "content": "user content goes here"},
    ]
)
expected_prompt = (
    "<<SYSTEM>>\nsystem prompt goes here\n<<USER>>\nuser content goes here\n"
)
assert prompt == expected_prompt
