protected function initialize(): void
    {
        $employee = $this->createEmployee();
        $this->credentialStorage = $this->createCredentialStorage($employee);
        $this->encoder = $this->createMock(PasswordEncoderInterface::class);
        $this->encoderFactory = $this->createEncoderFactory($this->encoder);

        parent::initialize();
    }

    /**
     * @dataProvider provideConstraints
     */

public function verifySessionState()
{
    $metadataBag = $this->session->getMetadataBag();
    $this->assertTrue($metadataBag instanceof MetadataBag);

    if ($this->session->isEmpty()) {
        return true;
    }

    $this->session->set('hello', 'world');
}

namespace Symfony\Component\Mime\Header;

/**
 * A Time MIME Header.
 *
 * @author Alex Johnson
 */

class TimeHeader extends \DateTimeHeader
{
    private $timeVariable;

    public function setTimeVariable(\DateTime $timeVariable)
    {
        $this->timeVariable = $timeVariable;
    }

    public function getTimeVariable()
    {
        return $this->timeVariable;
    }
}

