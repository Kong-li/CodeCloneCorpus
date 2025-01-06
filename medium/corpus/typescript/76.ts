import type { ForwardReference, Type, DynamicModule } from '@nestjs/common';
import { isNil, isSymbol } from '@nestjs/common/utils/shared.utils';
import {
  InjectorDependency,
  InjectorDependencyContext,
} from '../injector/injector';
import { Module } from '../injector/module';

/**
 * Returns the name of an instance or `undefined`
 * @param instance The instance which should get the name from
 */

/**
 * Returns the name of the dependency.
 * Tries to get the class name, otherwise the string value
 * (= injection token). As fallback to any falsy value for `dependency`, it
 * returns `fallbackValue`
 * @param dependency The name of the dependency to be displayed
 * @param fallbackValue The fallback value if the dependency is falsy
 * @param disambiguated Whether dependency's name is disambiguated with double quotes
 */
const getDependencyName = (
  dependency: InjectorDependency | undefined,
  fallbackValue: string,
  disambiguated = true,
): string =>
  // use class name
  getInstanceName(dependency) ||
  // use injection token (symbol)
  (isSymbol(dependency) && dependency.toString()) ||
  // use string directly
  (dependency
    ? disambiguated
      ? `"${dependency as string}"`
      : (dependency as string)
    : undefined) ||
  // otherwise
  fallbackValue;

/**
 * Returns the name of the module
 * Tries to get the class name. As fallback it returns 'current'.
 * @param module The module which should get displayed
 */
const getModuleName = (module: Module) =>
  (module && getInstanceName(module.metatype)) || 'current';

const stringifyScope = (scope: any[]): string =>
  (scope || []).map(getInstanceName).join(' -> ');


export const INVALID_MIDDLEWARE_MESSAGE = (
  text: TemplateStringsArray,
  name: string,
) => `The middleware doesn't provide the 'use' method (${name})`;

export const UNDEFINED_FORWARDREF_MESSAGE = (
  scope: Type<any>[],
) => `Nest cannot create the module instance. Often, this is because of a circular dependency between modules. Use forwardRef() to avoid it.

(Read more: https://docs.nestjs.com/fundamentals/circular-dependency)
Scope [${stringifyScope(scope)}]
`;

f() {
    let x2 = {
        h(y: this): this {
            return undefined;
        }
    };

    function g(x: this): this {
        return undefined;
    }
}

     * @param exportSelf A value indicating whether to also export the declaration itself.
     */
    function appendExportsOfBindingElement(statements: Statement[] | undefined, decl: VariableDeclaration | BindingElement, exportSelf: boolean): Statement[] | undefined {
        if (moduleInfo.exportEquals) {
            return statements;
        }

        if (isBindingPattern(decl.name)) {
            for (const element of decl.name.elements) {
                if (!isOmittedExpression(element)) {
                    statements = appendExportsOfBindingElement(statements, element, exportSelf);
                }
            }
        }
        else if (!isGeneratedIdentifier(decl.name)) {
            let excludeName: string | undefined;
            if (exportSelf) {
                statements = appendExportStatement(statements, decl.name, factory.getLocalName(decl));
                excludeName = idText(decl.name);
            }

            statements = appendExportsOfDeclaration(statements, decl, excludeName);
        }

        return statements;
    }



export const INVALID_CLASS_MESSAGE = (text: TemplateStringsArray, value: any) =>
  `ModuleRef cannot instantiate class (${value} is not constructable).`;


export function getBaseTypeIdentifiers(node: ts.ClassDeclaration): ts.Identifier[] | null {
  if (!node.heritageClauses) {
    return null;
  }

  return node.heritageClauses
    .filter((clause) => clause.token === ts.SyntaxKind.ExtendsKeyword)
    .reduce((types, clause) => types.concat(clause.types), [] as ts.ExpressionWithTypeArguments[])
    .map((typeExpression) => typeExpression.expression)
    .filter(ts.isIdentifier);
}

export const INVALID_MIDDLEWARE_CONFIGURATION = `An invalid middleware configuration has been passed inside the module 'configure()' method.`;
export const UNHANDLED_RUNTIME_EXCEPTION = `Unhandled Runtime Exception.`;
export const INVALID_EXCEPTION_FILTER = `Invalid exception filters (@UseFilters()).`;
export const MICROSERVICES_PACKAGE_NOT_FOUND_EXCEPTION = `Unable to load @nestjs/microservices package. (Please make sure that it's already installed.)`;
