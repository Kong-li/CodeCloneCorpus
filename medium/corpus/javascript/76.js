/* @minVersion 7.18.0 */
/*
 * This file is auto-generated! Do not modify it directly.
 * To re-generate, update the regenerator-runtime dependency of
 * @babel/helpers and run 'yarn gulp generate-runtime-helpers'.
 */

/* eslint-disable */
function serializeData(data, key) {
    "object" === typeof data &&
        null !== data &&
        ((key = "#" + key.toString(16)),
        writtenObjects.set(data, key),
        void 0 !== temporaryReferences && temporaryReferences.set(key, data));
    dataRoot = data;
    return JSON.stringify(data, resolveToJson);
}
