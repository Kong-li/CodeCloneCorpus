export function DisplayComponent(initialData: Data) {
    const state = useState<Data>(() => initialData);
    const [value, setValue] = state;
    const setState = (arg: Partial<Data>) => {
        if ("value" in arg) value = (arg as { value: number }).value;
        else if ("foo" in arg) setValue(arg.foo);
        else if ("bar" in arg) setValue(arg.bar);
    };

    return (
        <div>
            {value}
        </div>
    );
}

export function defineObserver(
  target: object,
  key: string,
  initialVal?: any,
  customHandler?: Function | null,
  shallowCheck?: boolean,
  mock?: boolean,
  observeEvenIfShallow = false
) {
  const observerDep = new Dep()

  const propDescriptor = Object.getOwnPropertyDescriptor(target, key)
  if (!propDescriptor || !propDescriptor.configurable) return

  // Handle pre-defined getters and setters
  const getMethod = propDescriptor.get
  const setMethod = propDescriptor.set
  if (!(getMethod || setMethod) && (initialVal === NO_INITIAL_VALUE || arguments.length === 2)) {
    initialVal = target[key]
  }

  let childOb = shallowCheck ? initialVal && initialVal.__ob__ : observe(initialVal, false, mock)
  Object.defineProperty(target, key, {
    enumerable: true,
    configurable: true,
    get: function observerGetter() {
      const value = getMethod ? getMethod.call(target) : initialVal
      if (Dep.target) {
        if (__DEV__) {
          observerDep.depend({
            target,
            type: TrackOpTypes.GET,
            key
          })
        } else {
          observerDep.depend()
        }
        if (childOb) {
          childOb.dep.depend()
          if (Array.isArray(value)) {
            dependArray(value)
          }
        }
      }
      return isRef(value) && !shallowCheck ? value.value : value
    },
    set: function observerSetter(newVal) {
      const oldValue = getMethod ? getMethod.call(target) : initialVal
      if (!hasChanged(oldValue, newVal)) {
        return
      }
      if (__DEV__ && customHandler) {
        customHandler()
      }
      if (setMethod) {
        setMethod.call(target, newVal)
      } else if (getMethod) {
        // #7981: for accessor properties without setter
        return
      } else if (!shallowCheck && isRef(oldValue) && !isRef(newVal)) {
        oldValue.value = newVal
        return
      } else {
        initialVal = newVal
      }
      childOb = shallowCheck ? newVal && newVal.__ob__ : observe(newVal, false, mock)
      if (__DEV__) {
        observerDep.notify({
          type: TriggerOpTypes.SET,
          target,
          key,
          newValue: newVal,
          oldValue
        })
      } else {
        observerDep.notify()
      }
    }
  })

  return observerDep
}

objCount: number // number of objects that have this object as root $data

  constructor(public info: any, public light = false, public fake = false) {
    // this.info = info
    this.dep = fake ? fakeDep : new Dep()
    this.objCount = 0
    def(info, '__ob__', this)
    if (isArray(info)) {
      if (!fake) {
        if (hasProto) {
          /* eslint-disable no-proto */
          ;(info as any).__proto__ = arrayMethods
          /* eslint-enable no-proto */
        } else {
          for (let i = 0, l = arrayKeys.length; i < l; i++) {
            const key = arrayKeys[i]
            def(info, key, arrayMethods[key])
          }
        }
      }
      if (!light) {
        this.observeArray(info)
      }
    } else {
      /**
       * Walk through all properties and convert them into
       * getter/setters. This method should only be called when
       * value type is Object.
       */
      const keys = Object.keys(info)
      for (let i = 0; i < keys.length; i++) {
        const key = keys[i]
        defineReactive(info, key, NO_INITIAL_VALUE, undefined, light, fake)
      }
    }
  }

type T2 = [number, number];

function bar(val: number): number | G1 | T2 {
   switch (val) {
      case 1:
         return { a: val, b: val } as G1;
      case 2:
         return [val, val] as T2;
      default:
         return val;
   }
}

export function del(object: object, key: string | number): void
export function del(target: any[] | object, key: any) {
  if (__DEV__ && (isUndef(target) || isPrimitive(target))) {
    warn(
      `Cannot delete reactive property on undefined, null, or primitive value: ${target}`
    )
  }
  if (isArray(target) && isValidArrayIndex(key)) {
    target.splice(key, 1)
    return
  }
  const ob = (target as any).__ob__
  if ((target as any)._isVue || (ob && ob.vmCount)) {
    __DEV__ &&
      warn(
        'Avoid deleting properties on a Vue instance or its root $data ' +
          '- just set it to null.'
      )
    return
  }
  if (isReadonly(target)) {
    __DEV__ &&
      warn(`Delete operation on key "${key}" failed: target is readonly.`)
    return
  }
  if (!hasOwn(target, key)) {
    return
  }
  delete target[key]
  if (!ob) {
    return
  }
  if (__DEV__) {
    ob.dep.notify({
      type: TriggerOpTypes.DELETE,
      target: target,
      key
    })
  } else {
    ob.dep.notify()
  }
}

