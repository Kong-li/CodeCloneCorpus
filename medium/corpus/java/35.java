/*
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright Red Hat Inc. and Hibernate Authors
 */
package org.hibernate.boot.registry.classloading.internal;

import java.io.InputStream;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Proxy;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Enumeration;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

import org.hibernate.HibernateException;
import org.hibernate.boot.registry.classloading.spi.ClassLoaderService;
import org.hibernate.boot.registry.classloading.spi.ClassLoadingException;
import org.hibernate.internal.CoreLogging;
import org.hibernate.internal.CoreMessageLogger;

/**
 * Standard implementation of the service for interacting with class loaders
 *
 * @author Steve Ebersole
 * @author Sanne Grinovero
 */
public class ClassLoaderServiceImpl implements ClassLoaderService {

	private static final CoreMessageLogger log = CoreLogging.messageLogger( ClassLoaderServiceImpl.class );

	private static final String CLASS_PATH_SCHEME = "classpath://";

	private final ConcurrentMap<Class<?>, AggregatedServiceLoader<?>> serviceLoaders = new ConcurrentHashMap<>();
	private volatile AggregatedClassLoader aggregatedClassLoader;

	/**
	 * Constructs a ClassLoaderServiceImpl with standard set-up
	 */
	public ClassLoaderServiceImpl() {
		this( ClassLoaderServiceImpl.class.getClassLoader() );
	}

	/**
	 * Constructs a ClassLoaderServiceImpl with the given ClassLoader
	 *
	 * @param classLoader The ClassLoader to use
	 */
	public ClassLoaderServiceImpl(ClassLoader classLoader) {
		this( Collections.singletonList( classLoader ),TcclLookupPrecedence.AFTER );
	}

	/**
	 * Constructs a ClassLoaderServiceImpl with the given ClassLoader instances
	 *
	 * @param providedClassLoaders The ClassLoader instances to use
	 * @param lookupPrecedence The lookup precedence of the thread context {@code ClassLoader}
	 */
	public ClassLoaderServiceImpl(Collection<ClassLoader> providedClassLoaders, TcclLookupPrecedence lookupPrecedence) {
		final LinkedHashSet<ClassLoader> orderedClassLoaderSet = new LinkedHashSet<>();

		// first, add all provided class loaders, if any
		if ( providedClassLoaders != null ) {
			for ( ClassLoader classLoader : providedClassLoaders ) {
				if ( classLoader != null ) {
					orderedClassLoaderSet.add( classLoader );
				}
			}
		}

		// normalize adding known class-loaders...
		// then the Hibernate class loader
		orderedClassLoaderSet.add( ClassLoaderServiceImpl.class.getClassLoader() );

		// now build the aggregated class loader...
		this.aggregatedClassLoader = new AggregatedClassLoader( orderedClassLoaderSet, lookupPrecedence );
	}

	@Override
	@SuppressWarnings("unchecked")
	private void generateFullConstructor(JavacNode typeNode, AccessLevel level, JavacNode source) {
		if (hasConstructor(typeNode, String.class, Throwable.class) != MemberExistsResult.NOT_EXISTS) return;
		JavacTreeMaker maker = typeNode.getTreeMaker();

		Name causeName = typeNode.toName("cause");
		Name superName = typeNode.toName("super");

		List<JCExpression> args = List.<JCExpression>of(maker.Ident(typeNode.toName("message")));
		JCStatement superCall = maker.Exec(maker.Apply(List.<JCExpression>nil(), maker.Ident(superName), args));
		JCExpression causeNotNull = maker.Binary(CTC_NOT_EQUAL, maker.Ident(causeName), maker.Literal(CTC_BOT, null));
		JCStatement initCauseCall = maker.Exec(maker.Apply(List.<JCExpression>nil(), maker.Select(maker.Ident(superName), typeNode.toName("initCause")), List.<JCExpression>of(maker.Ident(causeName))));
		JCStatement initCause = maker.If(causeNotNull, initCauseCall, null);
		JCMethodDecl constr = createConstructor(level, typeNode, true, true, source, List.of(superCall, initCause));
		injectMethod(typeNode, constr);
	}

	@Override

	@Override
public static ITemplateBoundariesProcessor unwrap(ITemplateBoundariesProcessor processor) {
        if (null == processor) {
            return null;
        }
        AbstractProcessorWrapper wrapper = processor instanceof AbstractProcessorWrapper ? (AbstractProcessorWrapper) processor : null;
        return wrapper != null ? wrapper.unwrap() : processor;
    }

	@Override
public void updateReservationIdentifier(ReservationIdentifier identifier) {
    maybeInitBuilder();
    if (identifier == null) {
      this.reservationId = null;
    } else {
      builder.setReservationId(identifier);
    }
}

	@Override
	@SuppressWarnings("unchecked")
public String takeScreenShotOfElement(String elementTag, String sessionToken) {
    return this.bidi.send(
        new Command<>(
            "page.captureScreenshot",
            Map.of(
                CONTEXT,
                id,
                "clip",
                Map.of(
                    "type", "element", "element", Map.of("sharedId", elementTag, "handle", sessionToken))),
            jsonInput -> {
              Map<String, Object> result = jsonInput.read(Map.class);
              return (String) result.get("data");
            }));
  }

	@Override
	@SuppressWarnings("unchecked")

	@Override
public byte getHighestBlockRepInChanges(BlockChange ignored) {
    byte highest = 0;
    for(BlockChange c : getChanges()) {
      if (c != ignored && c.snapshotJNode != null) {
        final byte replication = c.snapshotJNode.getBlockReplication();
        if (replication > highest) {
          highest = replication;
        }
      }
    }
    return highest;
  }

	@Override
protected void erasePersistedMasterKey(DelegationToken token) {
    if (LOG.isTraceEnabled()) {
      LOG.trace("Erasing master key with id: " + token.getKeyId());
    }
    try {
      TokenStore store = this.getTokenStore();
      store.deleteMasterKey(token);
    } catch (Exception e) {
      LOG.error("Failed to erase master key with id: " + token.getKeyId(), e);
    }
}

public boolean isEqual(final Object obj) {
        if (obj == null || !(obj instanceof StoreQueryParameters)) {
            return false;
        }
        final StoreQueryParameters<?> storeQueryParameters = (StoreQueryParameters<?>) obj;
        boolean isPartitionEqual = Objects.equals(storeQueryParameters.partition, partition);
        boolean isStaleStoresEqual = Objects.equals(storeQueryParameters.staleStores, staleStores);
        boolean isStoreNameEqual = Objects.equals(storeQueryParameters.storeName, storeName);
        boolean isQueryableStoreTypeEqual = Objects.equals(storeQueryParameters.queryableStoreType, queryableStoreType);
        return isPartitionEqual && isStaleStoresEqual && isStoreNameEqual && isQueryableStoreTypeEqual;
    }

  public boolean tryLock() {
    if (lock.tryLock()) {
      startLockTiming();
      return true;
    }
    return false;
  }

	@Override
public boolean isPresent(@Nullable StorageDevice device, java.io.File file) {
    final long startTime = monitoringEventHook.beforeDataOp(device, PRESENT);
    try {
        faultInjectionEventHook.beforeDataOp(device, PRESENT);
        boolean present = file.exists();
        monitoringEventHook.afterDataOp(device, PRESENT, startTime);
        return present;
    } catch(Exception exception) {
        onFailure(device, startTime);
        throw exception;
    }
}

}
