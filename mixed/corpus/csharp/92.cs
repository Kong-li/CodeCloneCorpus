public static IApplicationBuilder ApplyMvcRoutes(this IApplicationBuilder applicationBuilder)
{
    ArgumentNullException.ThrowIfNull(applicationBuilder);

    applicationBuilder.UseMvc(routes =>
    {
        // 保持空实现
    });

    return applicationBuilder;
}

else if (status != null)
        {
            writer.Append(status);

            if (error != null)
            {
                writer
                    .AppendLine()
                    .Append(error);
            }
        }

public static void ProcessJsonProperties(JsonReader reader, Action<string> propertyCallback)
    {
        while (reader.Read())
        {
            if (reader.TokenType == JsonToken.PropertyName)
            {
                string propertyName = reader.Value.ToString();
                propertyCallback(propertyName);
            }
            else if (reader.TokenType == JsonToken.EndObject)
            {
                break;
            }
        }
    }

