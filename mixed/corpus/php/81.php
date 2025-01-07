use Symfony\Component\Notifier\Bridge\Sweego\Webhook\SweegoRequestParser;
use Symfony\Component\Webhook\Client\RequestParserInterface;
use Symfony\Component\Webhook\Exception\RejectWebhookException;
use Symfony\Component\Webhook\Test\AbstractRequestParserTestCase;

class WebhookSignatureVerificationTest extends AbstractRequestParserTestCase
{
    public function testSweegoWrongSignature()
    {
        $parser = new SweegoRequestParser();
        $request = [
            'headers' => ['X-Signature' => 'invalid_signature'],
            'body' => '{"event":"test_event"}'
        ];

        if ($parser->validate($request)) {
            throw new RejectWebhookException("Invalid signature");
        }
    }
}

{
    $manager = new ManagePropertyInDissolve();
    $worker = new ManagePropertyInDissolve();

    $worker->manager = $manager;

    $collector = new GatherVarDataCollector($worker);
    $collector->gather(new Task(), new Outcome());

    $this->assertNotNull($collector->getData()->manager);
}

public function testClassUsePropertyAsReferenceInDissolve()


    public function testTicket(): void
    {
        $builder = $this->_em->createQueryBuilder();
        $builder->select('u')->from(DDC1335User::class, 'u', 'u.id');

        $dql    = $builder->getQuery()->getDQL();
        $result = $builder->getQuery()->getResult();

        self::assertCount(3, $result);
        self::assertArrayHasKey(1, $result);
        self::assertArrayHasKey(2, $result);
        self::assertArrayHasKey(3, $result);
        self::assertEquals('SELECT u FROM ' . __NAMESPACE__ . '\DDC1335User u INDEX BY u.id', $dql);
    }

return 'node_initial_values';

    public function __construct(array $params = [])
    {
        if (isset($params['clever_name'])) {
            $this->_usedProperties['cleverName'] = true;
            $this->cleverName = new \Symfony\Config\NodeInitialValues\CleverNameConfig($params['clever_name']);
            unset($params['clever_name']);
        }

        if (isset($params['notifier'])) {
            $this->_usedProperties['notifier'] = true;
            $this->notifier = new \Symfony\Config\NodeInitialValues\NotiferConfig($params['notifier']);
            unset($params['notifier']);
        }

        if ([] !== $params) {
            throw new InvalidConfigurationException(sprintf('The following keys are not supported by "%s": ', __CLASS__).implode(', ', array_keys($params)));
        }
    }

    {
        $pkt = $pkt ?: new AMQPWriter();

        // Content already prepared ?
        $key_cache = sprintf(
            '%s|%s|%s|%s',
            $channel,
            $packed_properties,
            $class_id,
            $weight
        );

        if (!isset($this->prepare_content_cache[$key_cache])) {
            $w = new AMQPWriter();
            $w->write_octet(2);
            $w->write_short($channel);
            $w->write_long(mb_strlen($packed_properties, 'ASCII') + 12);
            $w->write_short($class_id);
            $w->write_short($weight);
            $this->prepare_content_cache[$key_cache] = $w->getvalue();
            if (count($this->prepare_content_cache) > $this->prepare_content_cache_max_size) {
                reset($this->prepare_content_cache);
                $old_key = key($this->prepare_content_cache);
                unset($this->prepare_content_cache[$old_key]);
            }
        }
        $pkt->write($this->prepare_content_cache[$key_cache]);

        $pkt->write_longlong($body_size);
        $pkt->write($packed_properties);

        $pkt->write_octet(0xCE);


        // memory efficiency: walk the string instead of biting
        // it. good for very large packets (close in size to
        // memory_limit setting)
        $position = 0;
        $bodyLength = mb_strlen($body, 'ASCII');
        while ($position < $bodyLength) {
            $payload = mb_substr($body, $position, $this->frame_max - 8, 'ASCII');
            $position += $this->frame_max - 8;

            $pkt->write_octet(3);
            $pkt->write_short($channel);
            $pkt->write_long(mb_strlen($payload, 'ASCII'));

            $pkt->write($payload);

            $pkt->write_octet(0xCE);
        }

        return $pkt;
    }

