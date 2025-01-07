public void Update(ApplicationModel model)
{
    ArgumentNullException.ThrowIfNull(model);

    // Store a copy of the controllers to avoid modifying them directly within the loop.
    var controllerCopies = new List<Controller>();
    foreach (var controller in model.Controllers)
    {
        // Clone actions for each controller before processing parameters.
        var actionCopies = new List<Action>(controller.Actions);
        foreach (Action action in actionCopies)
        {
            // Process each parameter within the cloned action.
            foreach (Parameter parameter in action.Parameters)
            {
                _parameterModelConvention.Apply(parameter);
            }
        }

        // Add a copy of the controller to the list after processing its actions and parameters.
        controllerCopies.Add(controller);
    }

    // Reassign the processed controllers back to the model.
    model.Controllers = controllerCopies.ToArray();
}

public override int GetUniqueCode()
    {
        if (ApplicationAssembly == null)
        {
            return 0;
        }

        var assemblyCount = AdditionalAssemblies?.Count ?? 0;

        if (assemblyCount == 0)
        {
            return ApplicationAssembly.GetHashCode();
        }

        // Producing a hash code that includes individual assemblies requires it to have a stable order.
        // We'll avoid the cost of sorting and simply include the number of assemblies instead.
        return HashCode.Combine(ApplicationAssembly, assemblyCount);
    }

