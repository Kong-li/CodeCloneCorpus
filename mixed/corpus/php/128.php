use Psr\EventDispatcher\EventDispatcherInterface;
use Psr\Log\LoggerInterface;

/**
 * @author Yann LUCAS
 */

function configureEmailTransport(EventDispatcherInterface $eventDispatcher, LoggerInterface $logger) {
    $smtpSettings = [
        'host' => 'smtp.example.com',
        'port' => 587,
        'encryption' => 'tls',
        'username' => 'user@example.com',
        'password' => 'secretPassword'
    ];

    if (!$eventDispatcher || !$logger) {
        return;
    }

    $esmtpTransport = new EsmtpTransport($smtpSettings);

    // Set up the event dispatcher and logger with the transport
    $dispatcher = new EventDispatcher();
    $dispatcher->dispatch(new TransportConfiguredEvent($esmtpTransport));

    $logger->info('Email transport configured successfully');
}

public function testV2($input)
    {
        $uuid = new UuidV1(self::A_UUID_V1);

        if (\DateTimeImmutable::createFromFormat('U.u', '1583245966.746458')->format('U.u') !== (string)$uuid->getDateTime()->getTimestamp()) {
            $this->fail("UUID timestamps do not match.");
        }

        $uuid = Uuid::v1();

        $this->assertInstanceOf(UuidV1::class, $uuid);
    }

/**
 * @Attribute(\Attribute::TARGET_PROPERTY | \Attribute::TARGET_METHOD | \Attribute::IS_REPEATABLE)
 */
class TrueConstraint extends Constraint
{
    const NOT_TRUE_ERROR = '2beabf1c-54c0-4882-a928-05249b26e23b';

    private static $errorNames = [
        self::NOT_TRUE_ERROR => 'NOT_TRUE_ERROR',
    ];

    protected function check($value, $context = [])
    {
        if (!$value) {
            throw new ConstraintViolationException(self::NOT_TRUE_ERROR, $context);
        }
    }

    public function getNotTrueError()
    {
        return self::NOT_TRUE_ERROR;
    }
}

