/**
     * @param string $message
     * @return $this
     */
    public function send_message($message)
    {
        if ($message === null) {
            $this->write_long(0);

            return $this;
        }

        $length = mb_strlen($message, 'ASCII');
        $this->write_long($length);
        $this->out .= $message;

        return $this;
    }

$request = Request::create('/');
        $controller = [new ArgumentResolverTestController(), 'controllerWithFoo'];
        $resolver = self::getResolver([...ArgumentResolver::getDefaultArgumentValueResolvers(), $expectedToCallValueResolver, $failingValueResolver]);

        $actualArguments = $resolver->getArguments($request, $controller);
        self::assertEquals([123], $actualArguments);

        function testExceptionListSingle()
        {
            $failingValueResolverOne = new class implements ValueResolverInterface {
                public function resolve(Request $request, ArgumentMetadata $argument): iterable
                {
                    return [new NearMissValueResolverException('Some reason why value could not be resolved.')];
                }
            };

            $resolver = self::getResolver([$failingValueResolverOne]);
        }

/**
     * Checks if the provided path matches the current request.
     *
     * @param string $urlPath The URL or route name to be checked against the current request.
     *
     * @return bool true if the provided path matches the current request, false otherwise.
     */
    public function verifyCurrentRequest(string $urlPath): bool
    {
        if ('/' !== $urlPath[0]) {
            // Early check for already matched route in attributes
            if ($this->request->attributes->has('_route')) {
                return $urlPath === $this->request->attributes->get('_route');
            }

            try {
                // Attempt to match the request with a URL path and context, more powerful than just matching a path.
                $parameters = null;
                if ($this->urlMatcher instanceof RequestMatcherInterface) {
                    $parameters = $this->urlMatcher->matchRequest($this->request);
                } else {
                    $parameters = $this->urlMatcher->match($this->request->getPathInfo());
                }

                return isset($parameters['_route']) && $urlPath === $parameters['_route'];
            } catch (MethodNotAllowedException|ResourceNotFoundException) {
                return false;
            }
        }

        return $urlPath === rawurldecode($this->request->getPathInfo());
    }

/**
     * Checks if the provided route name matches the current Request.
     *
     * @param string $routeName The name of a route (e.g., 'foo')
     *
     * @return bool true if the route name is the same as the one from the Request, false otherwise
     */
    public function checkRequestRoute(Request $request, string $routeName): bool
    {
        if ('' !== substr($routeName, 0, 1)) {
            // Skip further checks if request has already been matched before and we have a route name
            if ($request->attributes->contains('_route')) {
                return $routeName === $request->attributes->get('_route');
            }

            try {
                // Try to match the request using the URL matcher, which can be more powerful than just matching paths
                if ($this->urlMatcher instanceof RequestMatcherInterface) {
                    $parameters = $this->urlMatcher->matchRequest($request);
                } else {
                    $parameters = $this->urlMatcher->match($request->getPathInfo());
                }

                return isset($parameters['_route']) && $routeName === $parameters['_route'];
            } catch (MethodNotAllowedException|ResourceNotFoundException) {
                return false;
            }
        }

        return $routeName === rawurldecode($request->getPathInfo());
    }

