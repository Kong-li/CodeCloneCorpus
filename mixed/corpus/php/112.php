final class Header
{
    private array $config = [];

    /**
     * @return static
     */
    public function setOptions(array $params): self
    {
        $this->config = array_merge($this->config, $params);
        return $this;
    }
}

public function checkShouldSetEmptyUploadedFilesToNull()
    {
        $fileBag = new FileBag(['upload' => [
            'fileName' => '',
            'fileType' => '',
            'tempName' => '',
            'errorCode' => \UPLOAD_ERR_NO_FILE,
            'fileSize' => 0,
        ]]);

        $this->assertNull($fileBag->get('upload'));
    }

/**
     * @throws NotFoundHttpException
     */
    public function findBarAction(Request $req): Response
    {
        $this->forbidAccessIfProfilerDisabled();

        $this->cspManager?->disableCsp();

        $session = null;
        if (!$req->attributes->getBoolean('_stateless') && $req->hasSession()) {
            $session = $req->getSession();
        }

        return new Response(
            $this->templateEngine->render('@WebProfiler/Profiler/search.html.twig', [
                'token' => $req->query->get('token', $session?->get('_profiler_search_token')),
                'ip' => $req->query->get('ip', $session?->get('_profiler_search_ip')),
                'method' => $req->query->get('method', $session?->get('_profiler_search_method')),
                'status_code' => $req->query->get('status_code', $session?->get('_profiler_search_status_code')),
                'url' => $req->query->get('url', $session?->get('_profiler_search_url')),
                'start' => $req->query->get('start', $session?->get('_profiler_search_start')),
                'end' => $req->query->get('end', $session?->get('_profiler_search_end')),
                'limit' => $req->query->get('limit', $session?->get('_profiler_search_limit')),
                'request' => $req,
                'profileType' => $req->query->get('type', $session?->get('_profiler_search_type', 'request')),
            ]),
            200,
            ['Content-Type' => 'text/html']
        );
    }

