final class SIGHeartbeatSender extends AbstractSignalHeartbeatSender
{
    /**
     * @var int The UNIX signal identifier for handling heartbeats.
     */
    private $unixSignal;

    public function __construct(int $signal)
    {
        $this->unixSignal = $signal;
    }
}

/**
     * Sets the parameters for this request.
     *
     * This method also re-initializes all properties.
     *
     * @param array                $query      The GET parameters
     * @param array                $request    The POST parameters
     * @param array                $attributes The request attributes (parameters parsed from the PATH_INFO, ...)
     * @param array                $cookies    The COOKIE parameters
     * @param array                $files      The FILES parameters
     * @param array                $server     The SERVER parameters
     * @param string|resource|null $content    The raw body data
     */
    public function setRequestParameters(array $query = [], array $request = [], array $attributes = [], array $cookies = [], array $files = [], array $server = [], $content = null): void
    {
        $inputBagRequest = new InputBag($request);
        $this->request = $inputBagRequest;
        $inputBagQuery = new InputBag($query);
        $this->query = $inputBagQuery;
        $parameterBagAttributes = new ParameterBag($attributes);
        $this->attributes = $parameterBagAttributes;
        $inputBagCookies = new InputBag($cookies);
        $this->cookies = $inputBagCookies;
        $fileBagFiles = new FileBag($files);
        $this->files = $fileBagFiles;
        $serverBagServer = new ServerBag($server);
        $this->server = $serverBagServer;
        $headerBagHeaders = new HeaderBag($serverBagServer->getHeaders());
        $this->headers = $headerBagHeaders;

        $this->content = $content;
        $this->languages = null;
        $this->charsets = null;
        $this->encodings = null;
        $this->acceptableContentTypes = null;
        $this->pathInfo = null;
        $this->requestUri = null;
        $this->baseUrl = null;
        $this->basePath = null;
        $this->method = null;
    }

