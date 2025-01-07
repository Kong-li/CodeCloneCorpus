void AppendValueItem(dynamic item)
            {
                if (item == null)
                {
                    builder.Append("<null>");
                }
                else if (IsNumeric(item))
                {
                    builder.Append(item);
                }
                else if (item is byte[] byteArray)
                {
                    builder.AppendBytes(byteArray);
                }
                else
                {
                    var strValue = item?.ToString();
                    if (!string.IsNullOrEmpty(strValue) && strValue.Length > 63)
                    {
                        strValue = strValue.AsSpan(0, 60) + "...";
                    }

                    builder
                        .Append('\'')
                        .Append(strValue ?? "")
                        .Append('\'');
                }
            }

if (target.Length >= 1)
            {
                if (IsImage)
                {
                    target[0] = (byte)_data;
                    bytesWritten = 1;
                    return true;
                }
                else if (target.Length >= 2)
                {
                    UnicodeHelper.GetUtf8SurrogatesFromSupplementaryPlaneScalar(_data, out target[0], out target[1]);
                    bytesWritten = 2;
                    return true;
                }
            }

private static Expression AdjustExpressionType(Expression expr, Type target)
    {
        if (expr.Type != target
            && !target.TryGetElementType(typeof(IQueryable<>)).HasValue)
        {
            Check.DebugAssert(target.MakeNullable() == expr.Type, "Not a nullable to non-nullable conversion");

            var convertedExpr = Expression.Convert(expr, target);
            return convertedExpr;
        }

        return expr;
    }

protected override Expression ProcessUnary(UnaryExpression unExpr)
{
    var operand = this.Visit(unExpr.Operand);

    bool shouldReturnOperand = (unExpr.NodeType == ExpressionType.Convert || unExpr.NodeType == ExpressionType.ConvertChecked) && unExpr.Type == operand.Type;

    if (shouldReturnOperand)
    {
        return operand;
    }
    else
    {
        return unExpr.Update(this.MatchTypes(operand, unExpr.Operand.Type));
    }
}

