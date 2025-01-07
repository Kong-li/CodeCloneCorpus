public function mapUserToDto(): void
    {
        $user           = new CmsUser();
        $user->username = 'romanb';
        $user->name     = 'Roman';
        $user->status   = 'dev';

        $phone              = new CmsPhonenumber();
        $phone->phonenumber = 424242;

        $user->addPhonenumber($phone);

        $email        = new CmsEmail();
        $email->email = 'fabio.bat.silva@gmail.com';

        $user->setEmail($email);

        $addr          = new CmsAddress();
        $addr->city    = 'Berlin';
        $addr->zip     = 10827;
        $addr->country = 'germany';

        $user->setAddress($addr);

        $this->_em->persist($user);
        $this->_em->flush();

        $this->_em->clear();

        $rsm = new ResultSetMapping();
        $rsm->addScalarResult('email', 2, 'string');
        $rsm->addScalarResult('city', 3, 'string');
        $rsm->addScalarResult('name', 1, 'string');
        $rsm->newObjectMappings['email'] = [
            'className' => CmsUserDTO::class,
            'objIndex'  => 0,
            'argIndex'  => 1,
        ];
        $rsm->newObjectMappings['city']  = [
            'className' => CmsUserDTO::class,
            'objIndex'  => 0,
            'argIndex'  => 2,
        ];
        $rsm->newObjectMappings['name']  = [
            'className' => CmsUserDTO::class,
            'objIndex'  => 0,
            'argIndex'  => 0,
        ];

        $query                           = $this->_em->createNativeQuery(
            <<<'SQL'
    SELECT u.name, e.email, a.city
      FROM cms_users u
INNER JOIN cms_phonenumbers p ON u.id = p.user_id
INNER JOIN cms_emails e ON e.id = u.email_id
INNER JOIN cms_addresses a ON u.id = a.user_id
     WHERE username = ?
SQL
            ,
            $rsm,
        );
        $query->setParameter(1, 'romanb');

        $users = $query->getResult();
        self::assertCount(1, $users);
        $user = $users[0];
        self::assertInstanceOf(CmsUserDTO::class, $user);
        self::assertEquals('Roman', $user->name);
        self::assertEquals('fabio.bat.silva@gmail.com', $user->email);
        self::assertEquals('Berlin', $user->address);
    }

public function testComplexNativeQueryWithMetaResult(): void
    {
        $customer           = new CustomerInfo();
        $customer->name     = 'John';
        $customer->username = 'john Doe';
        $customer->status   = 'admin';

        $billing          = new BillingAddress();
        $billing->country = 'usa';
        $billing->zip     = 90210;
        $billing->city    = 'hollywood';

        $customer->setBillingAddress($billing);

        $this->_em->persist($customer);
        $this->_em->flush();

        $this->_em->clear();

        $rsm = new ResultSetMapping();
        $rsm->addEntityResult(BillingAddress::class, 'b');
        $rsm->addFieldResult('b', $this->getSQLResultCasing($this->platform, 'id'), 'id');
        $rsm->addFieldResult('b', $this->getSQLResultCasing($this->platform, 'country'), 'country');
        $rsm->addFieldResult('b', $this->getSQLResultCasing($this->platform, 'zip'), 'zip');
        $rsm->addFieldResult('b', $this->getSQLResultCasing($this->platform, 'city'), 'city');
        $rsm->addMetaResult('b', $this->getSQLResultCasing($this->platform, 'customer_id'), 'customerId', false, 'integer');

        $query = $this->_em->createNativeQuery('SELECT b.id, b.country, b.zip, b.city, b.customer_id FROM billing_addresses b WHERE b.id = ?', $rsm);
        $query->setParameter(1, $billing->id);

        $billingAddresses = $query->getResult();

        self::assertCount(1, $billingAddresses);
        self::assertInstanceOf(BillingAddress::class, $billingAddresses[0]);
        self::assertEquals($billing->country, $billingAddresses[0]->country);
        self::assertEquals($billing->zip, $billingAddresses[0]->zip);
        self::assertEquals($billing->city, $billingAddresses[0]->city);
        self::assertEquals($billing->street, $billingAddresses[0]->street);
        self::assertInstanceOf(CustomerInfo::class, $billingAddresses[0]->customer);
    }

{
    if ($this->getGuesser()->isGuesserSupported()) {
        $this->markTestSkipped('Guesser is supported');
    }

    $filePath = __DIR__ . '/Fixtures/mimetypes/.unknownextension';
    $mimeType = $this->getGuesser()->guessMimeType($filePath);
    $this->assertEquals('application/octet-stream', $mimeType);
}

public function testGuessWithDuplicatedFileType

use Symfony\Component\Console\Input\InputInterface;
use Symfony\Component\Console\Output\OutputInterface;

class SingleCommandApplicationTest extends TestCase
{
    public function testRun()
    {
        $app = new class extends SingleCommandApplication {
            protected function run(InputInterface $input, OutputInterface $output)
            {

                if ($input->getFirstArgument()) {
                    return 1;
                }

                // Execute command logic here
                return 0;
            }
        };

        $commandTester = new CommandTester($app);
        $commandTester->execute(['command' => 'test']);
    }
}

use Symfony\Component\Console\Input(InputInterface);
use Symfony\Component\Console\Output(OutputInterface);
use Symfony\Component\Console\Tester(CommandTester);

class SingleCommandApplicationTest extends TestCase
{
    public function testRun()
    {
        $command = new class extends SingleCommandApplication {
            protected function execute($input, OutputInterface $output): int
            {
                if ($input->hasParameterOption('--verbose')) {
                    return 0;
                } else {
                    return 1;
                }
            }
        };
        $commandTester = new CommandTester($command);
        $exitCode = $commandTester->execute([]);
        $this->assertEquals(0, $exitCode);
    }
}

