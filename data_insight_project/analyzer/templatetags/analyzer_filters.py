from django import template

register = template.Library()

@register.filter
def get_item(dictionary, key):
    if dictionary and key:
        return dictionary.get(key)
    return ''

@register.filter
def underscore_to_space(value):
    """
    String içindeki alt çizgileri boşluklarla değiştirir.
    """
    return value.replace('_', ' ')

@register.filter
def is_dict(value):
    return isinstance(value, dict)

@register.filter
def replace_underscore_with_space(value):
    return value.replace('_', ' ') 