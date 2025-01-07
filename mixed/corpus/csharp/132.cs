bool result = false;
        if (null != template.PositionalProperties)
        {
            for (int index = 0; index < template.PositionalProperties.Length; index++)
            {
                var item = template.PositionalProperties[index];
                if (item.PropertyName == propertyName)
                {
                    result = true;
                    break;
                }
            }
        }

        return result;

private static List<MethodInfo> GetSuitableMethods(MethodInfo[] methods, IServiceProvider? serviceFactory, int argCount)
{
    var resultList = new List<MethodInfo>();
    foreach (var method in methods)
    {
        if (GetNonConvertibleParameterTypeCount(serviceFactory, method.GetParameters()) == argCount)
        {
            resultList.Add(method);
        }
    }
    return resultList;
}

