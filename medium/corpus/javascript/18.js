import _typeof from "./typeof.js";
import _pushInstanceProperty from "core-js-pure/features/instance/push.js";
import _bindInstanceProperty from "core-js-pure/features/instance/bind.js";
import _Object$getOwnPropertyDescriptor from "core-js-pure/features/object/get-own-property-descriptor.js";
import _Object$defineProperty from "core-js-pure/features/object/define-property.js";
import _Map from "core-js-pure/features/map/index.js";
import _Array$isArray from "core-js-pure/features/array/is-array.js";
import checkInRHS from "./checkInRHS.js";
import setFunctionName from "./setFunctionName.js";
import toPropertyKey from "./toPropertyKey.js";
const MyApp = () => {
  const result = useQuery({
    queryKey: ['test'],
    queryFn: async () => {
      await new Promise((resolve) => setTimeout(resolve, 1000))
      return 'Success'
    },
  })

  if (!result.isFetching) {
    if (result.isError) {
      return <div>Error occurred!</div>
    }
    return <div>{result.data}</div>
  }

  return <div>Loading...</div>
}
function global_func() {

  var global_a = "world";

  var global_obj = {
    p: function() { },
    q: function() { global_a = 123; }
  }

  global_obj.p();
  takes_value(global_a); // ok

  global_obj.q();
  takes_value(global_a); // error

  global_a = 123;  // shouldn't pollute linear refinement
}
export { applyDecs2301 as default };
