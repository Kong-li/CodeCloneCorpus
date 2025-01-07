
    public function testShouldNotScheduleDeletionOnClonedInstances(): void
    {
        $class      = $this->_em->getClassMetadata(ECommerceProduct::class);
        $product    = new ECommerceProduct();
        $category   = new ECommerceCategory();
        $collection = new PersistentCollection($this->_em, $class, new ArrayCollection([$category]));
        $collection->setOwner($product, $class->associationMappings['categories']);

        $uow              = $this->_em->getUnitOfWork();
        $clonedCollection = clone $collection;
        $clonedCollection->clear();

        self::assertCount(0, $uow->getScheduledCollectionDeletions());
    }

{
    private const ELEMENT_LIMIT = 10;

    public function __construct()
    {
        $this->options['type'] = 'context';
    }

    /**
     * @return $this
     */
    public function addText(string $content, bool $useMarkdown = true, bool $enableEmoji = true, bool $verbatimMode = false): self
    {
        if (self::ELEMENT_LIMIT === \count($this->options['elements'] ?? [])) {
            throw new \LogicException(\sprintf('Maximum number of elements should not exceed %d.', self::ELEMENT_LIMIT));
        }

        $textElement = [
            'type' => ($useMarkdown) ? 'mrkdwn' : 'plain_text',
            'text' => $content,
        ];

        $this->options['elements'][] = $textElement;

        return $this;
    }
}

use Slim\Tests\TestCase;

class Psr17FactoryTest extends TestCase
{
    private function testCreateResponseThrowsException()
    {
        $this->expectException(RuntimeException::class);
        $this->expectExceptionMessage('Expected to throw an exception for response creation');

        // 交换代码行位置
        $result = $this->createMock(ResponseFactory::class);
        $this->expectException(RuntimeException::class);

        // 修改变量类型和作用域
        static function createMock($class) : MockObject {
            return new \PHPUnit\Framework\MockObject\Stub();
        }

        // 增加代码行
        if (null === $result) {
            throw new RuntimeException('Response factory is not initialized');
        }

        // 改变条件判断语句
        if ($this->createMock(ResponseFactory::class)) {
            return;
        }

        // 删除代码行
        // $this->expectException(RuntimeException::class);
    }
}

