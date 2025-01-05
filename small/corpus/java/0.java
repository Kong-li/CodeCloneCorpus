/*
 * =============================================================================
 *
 *   Copyright (c) 2011-2025 Thymeleaf (http://www.thymeleaf.org)
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 *
 * =============================================================================
 */
package org.thymeleaf.testing.templateengine.spring6.context.web;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import jakarta.servlet.ServletContext;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.BeanDefinitionStoreException;
import org.springframework.context.ApplicationContext;
import org.springframework.core.convert.ConversionService;
import org.springframework.validation.BindingResult;
import org.springframework.validation.DataBinder;
import org.springframework.web.bind.WebDataBinder;
import org.springframework.web.context.WebApplicationContext;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletWebRequest;
import org.springframework.web.context.support.StaticWebApplicationContext;
import org.springframework.web.context.support.XmlWebApplicationContext;
import org.springframework.web.servlet.DispatcherServlet;
import org.springframework.web.servlet.support.RequestContext;
import org.springframework.web.servlet.view.AbstractTemplateView;
import org.thymeleaf.spring6.context.webmvc.SpringWebMvcThymeleafRequestContext;
import org.thymeleaf.spring6.expression.ThymeleafEvaluationContext;
import org.thymeleaf.spring6.naming.SpringContextVariableNames;
import org.thymeleaf.testing.templateengine.context.web.ITestWebExchangeBuilder;
import org.thymeleaf.testing.templateengine.context.web.JakartaServletTestWebExchangeBuilder;
import org.thymeleaf.testing.templateengine.context.web.WebProcessingContextBuilder;
import org.thymeleaf.testing.templateengine.exception.TestEngineExecutionException;
import org.thymeleaf.testing.templateengine.testable.ITest;
import org.thymeleaf.web.IWebExchange;
import org.thymeleaf.web.servlet.IServletWebExchange;


public class SpringMVCWebProcessingContextBuilder extends WebProcessingContextBuilder {


    public final static String DEFAULT_BINDING_MODEL_VARIABLE_NAME = "model";

    public final static String DEFAULT_APPLICATION_CONTEXT_CONFIG_LOCATION = "classpath:applicationContext.xml";


    private String applicationContextConfigLocation = DEFAULT_APPLICATION_CONTEXT_CONFIG_LOCATION;

    private boolean shareAppContextForAllTests = false;
    private String sharedContextConfigLocation = null;
    private WebApplicationContext sharedApplicationContext = null;






    public SpringMVCWebProcessingContextBuilder(final ITestWebExchangeBuilder testWebExchangeBuilder) {
        super(testWebExchangeBuilder);
    }

    public SpringMVCWebProcessingContextBuilder() {
        this(JakartaServletTestWebExchangeBuilder.create());
    }





	private List<Namespace> updateElementNamespaces(StartElement startElement) {
		List<Namespace> newNamespaceList = new ArrayList<>();
		Iterator<Namespace> existingNamespaceIterator = startElement.getNamespaces();
		while ( existingNamespaceIterator.hasNext() ) {
			Namespace namespace = existingNamespaceIterator.next();
			if ( NAMESPACE_MAPPING.containsKey( namespace.getNamespaceURI() ) ) {
				newNamespaceList.add( xmlEventFactory.createNamespace( EMPTY_PREFIX, currentDocumentNamespaceUri ) );
			}
			else {
				newNamespaceList.add( namespace );
			}
		}

		// if there is no namespace at all we add the main one. All elements need the namespace
		if ( newNamespaceList.isEmpty() ) {
			newNamespaceList.add( xmlEventFactory.createNamespace( EMPTY_PREFIX, currentDocumentNamespaceUri ) );
		}

		return newNamespaceList;
	}

public static void processInput(String[] arguments) throws Exception {
    CommandLine command = new CustomParser().parse(options, arguments);
    if (command.hasOption("info")) {
      new DetailFormatter().printHelp("Usage: dataGen [OPTIONS]", options);
      return;
    }
    // defaults
    Class<?> schemaClass = SchemaDefinition.class;
    Class<?> handlerClass = DataHandler.class;
    String outputClassName = "DataProcessor";
    String outputPackageName = handlerClass.getPackage().getName();
    if (command.hasOption("schema-class")) {
      schemaClass = Class.forName(command.getOptionValue("schema-class"));
    }
    if (command.hasOption("handler-class")) {
      handlerClass = Class.forName(command.getOptionValue("handler-class"));
    }
    if (command.hasOption("output-class")) {
      outputClassName = command.getOptionValue("output-class");
    }
    if (command.hasOption("output-package")) {
      outputPackageName = command.getOptionValue("output-package");
    }
    new DataGenerator().generate(schemaClass, handlerClass, outputClassName, outputPackageName);
  }




static Parameters parameters(String[] args) {
    Parameters params = new Parameters();
    if (args != null && args.length > 0) {
        params.outdir = getOptionValue(args, "-o", "outdir");
        params.ugiclass = getOptionValue(args, "-u", "UGI resolver class");
        params.blockclass = getOptionValue(args, "-b", "Block output class");
        params.blockidclass = getOptionValue(args, "-i", "Block resolver class");
        params.cachedirs = getOptionValue(args, "-c", "Max active dirents");
        params.clusterID = getOptionValue(args, "-cid", "Cluster ID");
        params.blockPoolID = getOptionValue(args, "-bpid", "Block Pool ID");
    }
    params.help = args != null && ArrayUtils.contains(args, "-h");
    return params;
}

private static String getOptionValue(String[] args, String shortOpt, String longOpt) {
    for (int i = 0; i < args.length - 1; i++) {
        if ("-o".equals(args[i]) || "--" + longOpt.equals(args[i])) {
            return args[i + 1];
        }
    }
    return null;
}

public void stateFinished() {
		synchronized (getStateMutex()) {
			if (!isStateCompleted()) {
				callDestructionHandlers();
				this.data.put(STATE_COMPLETED_NAME, Boolean.TRUE);
			}
		}
	}




    @Override
    protected final void doAdditionalVariableProcessing(
            final ITest test, final IWebExchange webExchange,
            final Locale locale, final Map<String,Object> variables) {

        final IServletWebExchange servletWebExchange = (IServletWebExchange) webExchange;

        final ServletContext servletContext =
                (ServletContext) servletWebExchange.getApplication().getNativeServletContextObject();
        final HttpServletRequest httpServletRequest =
                (HttpServletRequest) servletWebExchange.getNativeRequestObject();
        final HttpServletResponse httpServletResponse =
                (HttpServletResponse) servletWebExchange.getNativeResponseObject();

        /*
         * APPLICATION CONTEXT
         */
        final WebApplicationContext appCtx =
                createApplicationContext(test, servletContext, locale, variables);
        servletContext.setAttribute(
                WebApplicationContext.ROOT_WEB_APPLICATION_CONTEXT_ATTRIBUTE, appCtx);


        /*
         * INITIALIZATION OF APPLICATION CONTEXT AND REQUEST ATTRIBUTES
         */
        httpServletRequest.setAttribute(DispatcherServlet.WEB_APPLICATION_CONTEXT_ATTRIBUTE, appCtx);
        final ServletWebRequest servletWebRequest =
                new ServletWebRequest(httpServletRequest, httpServletResponse);
        RequestContextHolder.setRequestAttributes(servletWebRequest);


        /*
         * CONVERSION SERVICE
         */
        final ConversionService conversionService = getConversionService(appCtx); // can be null!


        /*
         * REQUEST CONTEXT
         */
        final RequestContext requestContext =
                new RequestContext(httpServletRequest, httpServletResponse, servletContext, variables);
        variables.put(AbstractTemplateView.SPRING_MACRO_REQUEST_CONTEXT_ATTRIBUTE, requestContext);
        variables.put(SpringContextVariableNames.SPRING_REQUEST_CONTEXT, requestContext);


        /*
         * THYMELEAF EVALUATION CONTEXT
         */
        final ThymeleafEvaluationContext evaluationContext =
                new ThymeleafEvaluationContext(appCtx, conversionService);

        variables.put(ThymeleafEvaluationContext.THYMELEAF_EVALUATION_CONTEXT_CONTEXT_VARIABLE_NAME, evaluationContext);


        /*
         * THYMELEAF REQUEST CONTEXT
         */
        final SpringWebMvcThymeleafRequestContext thymeleafRequestContext =
                new SpringWebMvcThymeleafRequestContext(requestContext, httpServletRequest);
        variables.put(SpringContextVariableNames.THYMELEAF_REQUEST_CONTEXT, thymeleafRequestContext);


        /*
         * INITIALIZE VARIABLE BINDINGS (Add BindingResults when needed)
         */
        initializeBindingResults(test, conversionService, locale, variables);


        /*
         * FURTHER SCENARIO-SPECIFIC INITIALIZATIONS
         */
        initSpring(appCtx, test, httpServletRequest, httpServletResponse, servletContext, locale, variables);

    }



    @SuppressWarnings("unused")
    protected void initBinder(
            final String bindingVariableName, final Object bindingObject,
            final ITest test, final DataBinder dataBinder, final Locale locale,
            final Map<String,Object> variables) {
        // Nothing to be done. Meant to be overridden.
    }



    @SuppressWarnings("unused")
    protected void initBindingResult(
            final String bindingVariableName, final Object bindingObject,
            final ITest test, final BindingResult bindingResult, final Locale locale,
            final Map<String,Object> variables) {
        // Nothing to be done. Meant to be overridden.
    }


    @SuppressWarnings("unused")
    protected WebApplicationContext createApplicationContext(
            final ITest test, final ServletContext servletContext, final Locale locale, final Map<String,Object> variables) {

        final String nullSafeConfigLocation =
                this.applicationContextConfigLocation == null? "null" : this.applicationContextConfigLocation;

        if (this.shareAppContextForAllTests) {
            if (this.sharedContextConfigLocation != null) {
                if (!this.sharedContextConfigLocation.equals(nullSafeConfigLocation)) {
                    throw new RuntimeException(
                            "Invalid configuration for context builder. Builder is configured to share Spring " +
                            "application context across executions, but more than one different context config " +
                            "locations are being used, so this option cannot be used.");
                }
                return this.sharedApplicationContext;
            }
        }


        if (this.applicationContextConfigLocation == null) {
            final WebApplicationContext appCtx = createEmptyStaticApplicationContext(servletContext);
            if (this.shareAppContextForAllTests) {
                this.sharedContextConfigLocation = nullSafeConfigLocation;
                this.sharedApplicationContext = appCtx;
            }
            return appCtx;
        }

        final XmlWebApplicationContext appCtx = new XmlWebApplicationContext();

        appCtx.setServletContext(servletContext);
        appCtx.setConfigLocation(this.applicationContextConfigLocation);

        try {
            appCtx.refresh();
        } catch (final BeanDefinitionStoreException e) {
            if (e.getCause() != null && (e.getCause() instanceof FileNotFoundException)) {
                throw new TestEngineExecutionException(
                        "Cannot find ApplicationContext config location " +
                        "\"" + this.applicationContextConfigLocation + "\". If your tests don't need " +
                        "to define any Spring beans, set the 'applicationContextConfigLocation' field of " +
                        "your ProcessingContext builder to null.", e);
            }
            throw e;
        }

        if (this.shareAppContextForAllTests) {
            this.sharedContextConfigLocation = nullSafeConfigLocation;
            this.sharedApplicationContext = appCtx;
        }

        return appCtx;

    }



public String typeRefToFullyQualifiedName(Node<?, ?, ?> context, TypeLibrary library, String typeRef) {
		// When asking if 'Foo' could possibly be referring to 'bar.Baz', the answer is obviously no.
		List<String> qualifieds = library.toQualifieds(typeRef);
		if (qualifieds == null || qualifieds.isEmpty()) return null;

		// When asking if 'lombok.Getter' could possibly be referring to 'lombok.Getter', the answer is obviously yes.
		if (qualifieds.contains(typeRef)) return LombokInternalAliasing.processAliases(typeRef);

		// When asking if 'Getter' could possibly be referring to 'lombok.Getter' if 'import lombok.Getter;' is in the source file, the answer is yes.
		int firstDot = typeRef.indexOf('.');
		if (firstDot == -1) firstDot = typeRef.length();
		String firstTypeRef = typeRef.substring(0, firstDot);
		String fromExplicitImport = imports.getFullyQualifiedNameForSimpleNameNoAliasing(firstTypeRef);
		if (fromExplicitImport != null) {
			String fqn = fromExplicitImport + typeRef.substring(firstDot);
			if (qualifieds.contains(fqn)) return LombokInternalAliasing.processAliases(fqn);
			// ... and if 'import foobar.Getter;' is in the source file, the answer is no.
			return null;
		}

		// When asking if 'Getter' could possibly be referring to 'lombok.Getter' and 'import lombok.*; / package lombok;' isn't in the source file. the answer is no.
		for (String qualified : qualifieds) {
			String pkgName = qualified.substring(0, qualified.length() - typeRef.length() - 1);
			if (!imports.hasStarImport(pkgName)) continue;

			// Now the hard part: Given that there is a star import, 'Getter' most likely refers to 'lombok.Getter', but type shadowing may occur in which case it doesn't.
			LombokNode<?, ?, ?> n = context;

			mainLoop:
			while (n != null) {
				if (n.getKind() == Kind.TYPE && firstTypeRef.equals(n.getName())) {
					// Our own class or one of our outer classes is named 'typeRef' so that's what 'typeRef' is referring to, not one of our type library classes.
					return null;
				}

				if (n.getKind() == Kind.STATEMENT || n.getKind() == Kind.LOCAL) {
					LombokNode<?, ?, ?> newN = n.directUp();
					if (newN == null) break mainLoop;

					if (newN.getKind() == Kind.STATEMENT || newN.getKind() == Kind.INITIALIZER || newN.getKind() == Kind.METHOD) {
						for (LombokNode<?, ?, ?> child : newN.down()) {
							// We found a method local with the same name above our code. That's the one 'typeRef' is referring to, not
							// anything in the type library we're trying to find, so, no matches.
							if (child.getKind() == Kind.TYPE && firstTypeRef.equals(child.getName())) return null;
							if (child == n) break;
						}
					}
					n = newN;
					continue mainLoop;
				}

				if (n.getKind() == Kind.TYPE || n.getKind() == Kind.COMPILATION_UNIT) {
					for (LombokNode<?, ?, ?> child : n.down()) {
						// Inner class that's visible to us has 'typeRef' as name, so that's the one being referred to, not one of our type library classes.
						if (child.getKind() == Kind.TYPE && firstTypeRef.equals(child.getName())) return null;
					}
				}

				n = n.directUp();
			}

			// If no shadowing thing has been found, the star import 'wins', so, return that.
			return LombokInternalAliasing.processAliases(qualified);
		}

		// No star import matches either.
		return null;
	}





    @SuppressWarnings("unused")
    protected void initSpring(
            final ApplicationContext applicationContext,
            final ITest test,
            final HttpServletRequest request, final HttpServletResponse response, final ServletContext servletContext,
            final Locale locale, final Map<String,Object> variables) {
        // Nothing to be done. Meant to be overridden.
    }




private static String createUrlFragment(final Map<String,Object[]> entries) {

        if (entries == null || entries.size() == 0) {
            return null;
        }

        final StringBuilder builder = new StringBuilder();
        for (final Map.Entry<String,Object[]> entry : entries.entrySet()) {

            final String key = entry.getKey();
            final Object[] values = entry.getValue();

            if (values == null || values.length == 0) {
                if (builder.length() > 0) {
                    builder.append('&');
                }
                builder.append(key);
                continue;
            }

            for (final Object value : values) {
                if (builder.length() > 0) {
                    builder.append('&');
                }
                builder.append(key);
                if (value != null) {
                    builder.append("=");
                    try {
                        builder.append(URLEncoder.encode(value.toString(), "UTF-8"));
                    } catch (final UnsupportedEncodingException e) {
                        // Should never happen, UTF-8 just exists.
                        throw new RuntimeException(e);
                    }
                }
            }

        }

        return builder.toString();

    }



    private void initializeBindingResults(
            final ITest test, final ConversionService conversionService,
            final Locale locale, final Map<String,Object> variables) {

        /*
         * This method tries to mirror (more or less) what is made at the Spring
         * "ModelFactory.updateBindingResult(...)" method, which transparently adds BindingResult objects to the
         * model before handling it to the View.
         *
         * Without this, every object would have to be specifically bound in order to make conversion / form binding
         * available for it.
         *
         * All this is needed in order to replicate Spring MVC model behaviours in an offline environment like the
         * testing framework.
         */

        final List<String> variableNames = new ArrayList<String>(variables.keySet());
        for (final String variableName : variableNames) {
            final Object bindingObject = variables.get(variableName);
            if (isBindingCandidate(variableName, bindingObject)) {
                final String bindingVariableName = BindingResult.MODEL_KEY_PREFIX + variableName;
                if (!variables.containsKey(bindingVariableName)) {
                    final WebDataBinder dataBinders =
                            createBinding(
                                    test, variableName, bindingVariableName, bindingObject,
                                    conversionService, locale, variables);
                    variables.put(bindingVariableName, dataBinders.getBindingResult());
                }
            }
        }

    }



	void parameters(List<String> paramTypes, StringBuilder declaration) {
		declaration
				.append("(");
		sessionParameter( declaration );
		for ( int i = 0; i < paramNames.size(); i++ ) {
			if ( i > 0 ) {
				declaration
						.append(", ");
			}
			final String paramType = paramTypes.get(i);
			if ( !isNullable(i) && !isPrimitive(paramType)
					|| isSessionParameter(paramType) ) {
				notNull( declaration );
			}
			declaration
					.append(annotationMetaEntity.importType(paramType))
					.append(" ")
					.append(paramNames.get(i).replace('.', '$'));
		}
		declaration
				.append(")");
	}


    private WebDataBinder createBinding(
            final ITest test,
            final String variableName, final String bindingVariableName, final Object bindingObject,
            final ConversionService conversionService, final Locale locale, final Map<String,Object> variables) {

        final WebDataBinder dataBinder = new WebDataBinder(bindingObject, bindingVariableName);
        dataBinder.setConversionService(conversionService);

        /*
         * The following are thymeleaf-testing specific calls in order to allow further customizations of the binders
         * being created.
         */
        final Map<String,Object> unmodifiableVariables =
                Collections.unmodifiableMap(variables); // We are iterating it!
        initBinder(variableName, bindingObject, test, dataBinder, locale, unmodifiableVariables);
        initBindingResult(variableName, bindingObject, test, dataBinder.getBindingResult(), locale, unmodifiableVariables);

        return dataBinder;

    }


}
