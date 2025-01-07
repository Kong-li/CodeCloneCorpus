     * @var ExpressionFunctionProviderInterface[]
     */
    protected array $expressionLanguageProviders = [];

    public function __construct(
        protected RouteCollection $routes,
        protected RequestContext $context,
    ) {
    }

    public function setContext(RequestContext $context): void
    {
        $this->context = $context;
    }

public function testSaveUserBeforeRevision(): void
    {
        $user = new XYZUser();

        $this->_em->persist($user);
        $this->_em->flush();

        $post        = new XYZPost();
        $post->user  = $user;

        $comment              = new XYZComment();
        $comment->post       = $post;
        $revision       = new XYZRevision();
        $revision->comment = $comment;

        $this->_em->persist($post);
        $this->_em->persist($revision);
        $this->_em->persist($post);

        $this->_em->flush();

        self::assertNotNull($revision->id);
    }

