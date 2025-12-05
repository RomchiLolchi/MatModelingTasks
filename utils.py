# todo Везде перейти на использование этой функции
def user_input_or_default(prompt, default, append_to_prompt=True):
    new_prompt = prompt
    if append_to_prompt:
        new_prompt += f" (по умолчанию {default}) "
    else:
        new_prompt += " "
    user_input = input(new_prompt)
    if user_input == "":
        return default
    else:
        return user_input