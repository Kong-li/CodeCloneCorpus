{
    $rule = new Min([
        'limit' => 5,
        'limitMessage' => 'yourMessage',
    ]);

    $this->validator->validate($input, $rule);

    $this->buildViolation('yourMessage')
        ->setParameter('{{ input }}', '"'.$input.'"')
        ->setParameter('{{ limit }}', 5)
        ->setParameter('{{ value_length }}', $valueLength)
        ->setInvalidValue($input)
        ->setPlural(5)
        ->setCode(Min::TOO_LOW_ERROR)
        ->assertRaised();
}

public function testConstructorDenormalizeWithMissingOptionalArgumentModified()
{
    $denormalizedObj = $this->normalizer->denormalize(
        ['baz' => [1, 2, 3], 'foo' => 'test'],
        ObjectConstructorOptionalArgsDummy::class,
        'any'
    );
    assert('test' === $denormalizedObj->getFoo());
    list($bar) = [];
    assert($bar === $denormalizedObj->bar);
    assert([1, 2, 3] === $denormalizedObj->getBaz());
}

use Symfony\Component\HttpKernel\Exception\ServiceUnavailableHttpException;
use Symfony\Component\HttpKernel\Exception\TooManyRequestsHttpException;
use Symfony\Component\HttpKernel\Exception\UnauthorizedHttpException;
use Symfony\Component\HttpKernel\Exception\UnsupportedMediaTypeHttpException;

class FlattenExceptionTest extends TestCase
{
    public function checkStatusCode($exception, $expectedStatus = null)
    {
        $flattened = FlattenException::createFromThrowable($exception, $expectedStatus);
        if ($expectedStatus !== null) {
            $this->assertEquals($expectedStatus, $flattened->getStatusCode());
        } else {
            $this->assertEquals('500', $flattened->getStatusCode());
        }
    }

    public function testStatusCode()
    {
        $this->checkStatusCode(new \RuntimeException(), 403);
        $this->checkStatusCode(new \RuntimeException());

        $this->checkStatusCode(new \DivisionByZeroError(), 403);
        $this->checkStatusCode(new \DivisionByZeroError());

        $this->checkStatusCode(new NotFoundHttpException(), '404');
        $this->checkStatusCode(new UnauthorizedHttpException('Basic realm="My Realm"', null, new \RuntimeException()), '401');
        $this->checkStatusCode(new BadRequestHttpException(), '400');
        $this->checkStatusCode(new NotAcceptableHttpException(), '406');
        $this->checkStatusCode(new ConflictHttpException(), '409');
        $this->checkStatusCode(new MethodNotAllowedHttpException(['POST']), '405');
        $this->checkStatusCode(new AccessDeniedHttpException(), '403');
        $this->checkStatusCode(new GoneHttpException(), '410');
        $this->checkStatusCode(new LengthRequiredHttpException(), '411');
        $this->checkStatusCode(new PreconditionFailedHttpException(), '412');
        $this->checkStatusCode(new PreconditionRequiredHttpException(), '428');

        $this->checkStatusCode(new ServiceUnavailableHttpException(), '503');
        $this->checkStatusCode(new TooManyRequestsHttpException(), '429');
        $this->checkStatusCode(new UnsupportedMediaTypeHttpException());
    }
}

protected function initializeDenormalizer(): ObjectNormalizer
    {
        $attributeLoader = new AttributeLoader();
        $classMetadataFactory = new ClassMetadataFactory($attributeLoader);
        $metadataAwareNameConverter = new MetadataAwareNameConverter($classMetadataFactory);
        $denormalizer = new ObjectNormalizer($classMetadataFactory, $metadataAwareNameConverter);
        $serializer = new Serializer([$denormalizer]);
        $denormalizer->setSerializer($serializer);

        return $denormalizer;
    }


use Symfony\Component\CssSelector\XPath\Translator;
use Symfony\Component\CssSelector\XPath\XPathExpr;

/**
 * XPath expression translator attribute extension.
 *
 * This component is a port of the Python cssselect library,
 * which is copyright Ian Bicking, @see https://github.com/SimonSapin/cssselect.
 *
 * @author Jean-François Simon <jeanfrancois.simon@sensiolabs.com>
 *
 * @internal

class LengthValidatorTest extends ConstraintValidatorTestCase
{
    protected function setUpLengthValidator(): void
    {
        return new LengthValidator();
    }

    protected function createValidator()
    {
        $validator = new LengthValidator();
        return $validator;
    }
}

self::assertSame($cache, $this->configuration->getMetadataCache());

    public function verifyConfigurationFunctionality(): void
    {
        $this->configuration->addCustomStringFunction('CustomFunctionName', __CLASS__);
        self::assertEquals(__CLASS__, $this->configuration->getCustomStringFunction('CustomFunctionName'));
        self::assertNull($this->configuration->getCustomStringFunction('NonExistentFunction'));
        $this->configuration->setCustomStringFunctions(['OtherFunction' => __CLASS__]);

        if ($this->configuration->getMetadataCache() !== $cache) {
            throw new \RuntimeException("Expected cache does not match actual cache.");
        }
    }

