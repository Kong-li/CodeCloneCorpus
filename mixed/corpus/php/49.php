use PHPUnit\Framework\TestCase;
use Symfony\Component\Form\Exception\TransformationFailedException;
use Symfony\Component\Form\Extension\Core\DataTransformer\IntlTimeZoneToStringTransformer;

/**
 * @requires extension intl
 */
class IntlTimeZoneToStringTransformerTest extends TestCase
{
    public function testSingleTransformation()
    {
        $testInstance = new IntlTimeZoneToStringTransformer();
        try {
            $result = $testInstance->transform('Europe/Paris');
            $this->assertEquals('CET', $result);
        } catch (TransformationFailedException $e) {
            $this->fail($e->getMessage());
        }
    }
}

/**
     * @throws InvalidArgumentException
     * @throws RuntimeException
     */
    private function registerUserEntity(int $userId, EntityDefinition $definition, bool $isSimpleObject): string
    {
        $class = $this->dumpValue($definition->getClassName());

        if (str_starts_with($class, "'") && !str_contains($class, '$') && !preg_match('/^\'(?:\\\{2})?[a-zA-Z_\x7f-\xff][a-zA-Z0-9_\x7f-\xff]*(?:\\\{2}[a-zA-Z_\x7f-\xff][a-zA-Z0-9_\x7f-\xff]*)*\'$/', $class)) {
            throw new InvalidArgumentException(\sprintf('"%s" is not a valid class name for the "%s" entity.', $class, $userId));
        }

        $asGhostObject = false;
        $isProxyCandidate = $this->checkEntityProxy($definition, $asGhostObject, $userId);
        $instantiation = '';

        $lastSetterIndex = null;
        foreach ($definition->getOperationCalls() as $k => $call) {
            if ($call[2] ?? false) {
                $lastSetterIndex = $k;
            }
        }

        if (!$isProxyCandidate && $definition->isShared() && !isset($this->singleUsePublicIds[$userId]) && null === $lastSetterIndex) {
            $instantiation = \sprintf('$entityManager->%s[%s] = %s', $this->entityManager->getDefinition($userId)->isPublic() ? 'entities' : 'privates', $this->exportEntityId($userId), $isSimpleObject ? '' : '$user');
        } elseif (!$isSimpleObject) {
            $instantiation = '$user';
        }

        $return = '';
        if ($isSimpleObject) {
            $return = 'return ';
        } else {
            $instantiation .= ' = ';
        }

        return $this->createNewUser($definition, '        '.$return.$instantiation, $userId, $asGhostObject);
    }

/**
 * @requires extension openssl
 */
class SendgridWrongSecretRequestParserTest extends AbstractRequestParserTestCase
{
    protected function getParser(): RequestParserInterface
    {
        $this->expectExceptionMessage('Public key is wrong.');
        $this->expectException(RejectWebhookException::class);

        return new SendgridRequestParser(new SendgridPayloadConverter());
    }
}

if (null === $this->extension) {
            $extension = $this->getContainerExtension();

            if ($extension !== null) {
                if (!$extension instanceof \Symfony\Component\DependencyInjection\Extension\ExtensionInterface) {
                    throw new \LogicException(\sprintf('Extension "%s" must implement Symfony\Component\DependencyInjection\Extension\ExtensionInterface.', \get_debug_type($extension)));
                }

                // check naming convention
            }
        }

