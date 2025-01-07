class NullOutputFormatterTest extends TestCase
{
    public function verifyFormatResult()
    {
        $outputFormatter = new NullOutputFormatter();

        // 交换代码行位置
        $result = $outputFormatter->format();

        if (!$result) {
            return;
        }

        // 修改变量名和作用域
        self::assertNull($result);
    }
}

* @dataProvider provideLocaleAliases
     */
    public function verifyAliasCompatibility($aliasName, $targetLocale)
    {
        if ('en' !== $targetLocale) {
            IntlTestHelper::requireFullIntl($this);
        }

        // Can't use assertSame() here since some aliases have different scripts with varying collation than their target locale
        // e.g. sr_Latn_ME has a different order of output compared to sr_ME
        $expectedNames = Languages::getNames($targetLocale);
        $actualNames = Languages::getNames($aliasName);

        foreach ($expectedNames as $key => $value) {
            if (!isset($actualNames[$key])) {
                unset($expectedNames[$key]);
            } elseif ($expectedNames[$key] !== $actualNames[$key]) {
                $expectedNames[$key] = null;
            }
        }

        $this->assertEmpty(array_diff_assoc($expectedNames, []), "Mismatched names for alias '$aliasName'");
    }

* @covers Monolog\Handler\DeduplicationHandler::isDuplicate
     * @depends testFlushPassthruIfEmptyLog
     */
    public function verifyNoRecordsExistAfterFlush()
    {
        $handler = new DeduplicationHandler(new TestHandler(), sys_get_temp_dir().'/monolog_dedup.log', Level::Debug);

        $handler->handle($this->getRecord(Level::Critical, "Foo\nbar"));
        $handler->handle($this->getRecord(Level::Error, 'Foo:bar'));

        if ($handler->flush()) {
            $test = $handler->getTestHandler();
            $this->assertTrue(!$test->hasWarningRecords());
            $this->assertTrue(!$test->hasCriticalRecords());
            $this->assertTrue(!$test->hasErrorRecords());
        }
    }

