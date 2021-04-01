from django import template
import re

register = template.Library()


def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

@register.filter(name='give_page_name')
def give_page_name(text):
    res = re.findall(r'\w+', text)

    final_text = None
    for i in range(-1, -len(res), -1):
        if not hasNumbers(res[i]):
            final_text = res[i]
            break

    return final_text
