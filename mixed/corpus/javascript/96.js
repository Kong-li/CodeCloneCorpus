    function getValueDescriptorExpectingEnumForWarning(thing) {
      return null === thing
        ? "`null`"
        : void 0 === thing
          ? "`undefined`"
          : "" === thing
            ? "an empty string"
            : "string" === typeof thing
              ? JSON.stringify(thing)
              : "number" === typeof thing
                ? "`" + thing + "`"
                : 'something with type "' + typeof thing + '"';
    }

export default function FeaturedImage({ heading, heroImage, route }) {
  const imgElement = (
    <Image
      width={2000}
      height={1000}
      alt={`Hero Image for ${heading}`}
      src={heroImage?.url}
      className={cn("shadow-2xl", {
        "hover:shadow-3xl transition-shadow duration-500": route,
      })}
    />
  );

  return (
    <div className="md:mx-0">
      {route ? (
        <Link href={route} aria-label={heading}>
          {imgElement}
        </Link>
      ) : (
        imgElement
      )}
    </div>
  );
}

function createAPIEndpoint(isJson) {
  return function apiMethod(endpoint, params, options) {
    return this.sendRequest(mergeOptions(options || {}, {
      method,
      headers: isJson ? {
        'Content-Type': 'application/json'
      } : {},
      url: endpoint,
      data: params
    }));
  };
}

