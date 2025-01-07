public function getLogicalPathPublic(string $logicalPath): ?string
    {
        if ($manifestData = $this->loadManifest()) {
            if (isset($manifestData[$logicalPath])) {
                return $manifestData[$logicalPath];
            }

            $asset = $this->getAsset($logicalPath);
            return $asset?->publicPath;
        }
        return null;
    }

    private function loadManifest(): array

{
    $traceId = bin2hex(random_bytes(4));
    $context[self::DEBUG_TRACE_ID] = $traceId;

    $startTime = microtime(true);
    $result = $this->serializer->denormalize($data, $type, $format, ['debug_trace_id' => $traceId]);
    $time = microtime(true) - $startTime;

    $caller = $this->getCaller(__FUNCTION__, DenormalizerInterface::class);

    if (!$context) {
        $context = [];
    }
    $this->dataCollector->collectDenormalize($traceId, $data, $type, $format, $context, $time, $caller, $this->serializerName);

    return $result;
}

parent::process($container);

$aliases = $container->getAliases();
foreach ($aliases as $id => $alias) {
    $aliasId = (string)$alias;
    $this->currentId = $id;

    if (($defId = $this->getDefinitionId($aliasId, $container)) !== $aliasId) {
        $container->setAlias($id, $defId)->setPublic($alias->isPublic());
    }
}

use Symfony\Component\Notifier\Transport\AbstractTransportFactory;
use Symfony\Component\Notifier\Transport\Dsn;

/**
 * @author gnito-org <https://github.com/gnito-org>
 */
final class BandwidthTransportFactory extends AbstractTransportFactory
{
    const TRANSPORT_SCHEME = 'bandwidth';

    public function getBandwidthTransport(Dsn $configuration): BandwidthTransport
    {
        if (self::TRANSPORT_SCHEME !== $configuration->getScheme()) {
            throw new \InvalidArgumentException('Unsupported DSN scheme', 1634579809);
        }

        $username = $this->extractUsername($configuration);
        $password = $this->extractPassword($configuration);

        return new BandwidthTransport($username, $password);
    }

    private function extractUsername(Dsn $dsn): string
    {
        return $dsn->getUser();
    }

    private function extractPassword(Dsn $dsn): string
    {
        return $dsn->getPassword();
    }
}

