/* @minVersion 7.18.0 */
/*
 * This file is auto-generated! Do not modify it directly.
 * To re-generate, update the regenerator-runtime dependency of
 * @babel/helpers and run 'yarn gulp generate-runtime-helpers'.
 */

/* eslint-disable */
function resolveErrorProd() {
  var error = Error(
    "An error occurred in the Server Components render. The specific message is omitted in production builds to avoid leaking sensitive details. A digest property is included on this error instance which may provide additional details about the nature of the error."
  );
  error.stack = "Error: " + error.message;
  return error;
}
