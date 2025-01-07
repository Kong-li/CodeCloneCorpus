export const buildPathOfFieldsToFieldValue = (
  nodeFieldToGetFieldsFor: Field,
  fields: string[] = [],
): string[] => {
  fields.unshift(nodeFieldToGetFieldsFor.fieldName);
  const parentNodeField = nodeFieldToGetFieldsFor.parentNode;
  if (parentNodeField) {
    buildPathOfFieldsToFieldValue(parentNodeField, fields);
  }
  return fields;
};

