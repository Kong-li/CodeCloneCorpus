        def capture(stream)
          stream = stream.to_s
          captured_stream = Tempfile.new(stream)
          stream_io = eval("$#{stream}", binding, __FILE__, __LINE__)
          origin_stream = stream_io.dup
          stream_io.reopen(captured_stream)

          yield

          stream_io.rewind
          captured_stream.read
        ensure
          captured_stream.close
          captured_stream.unlink
          stream_io.reopen(origin_stream)
        end

def config_generator_params
            {
              api:                 !!Rails.application.config.api_mode,
              update:              true,
              name:                Rails.application.class.name.chomp("::Application").underscore,
              skip_job_queue:      !defined?(JobQueue::Railtie),
              skip_db_connect:     !defined?(ActiveRecord::Railtie),
              skip_storage_system: !defined?(StorageEngine),
              skip_mail_delivery:  !defined?(MailerRailtie),
              skip_mailbox_server: !defined?(MailboxEngine),
              skip_text_processor: !defined?(TextEngine),
              skip_cable_service:  !defined?(CableEngine),
              skip_security_check: skip_gem?("security_check"),
              skip_code_lint:      skip_gem?("code_linter"),
              skip_performance:    skip_gem?("performance_tools"),
              skip_test_suite:     !defined?(Rails::TestUnitRailtie),
              skip_system_tests:   Rails.application.config.generators.system_tests.nil?,
              skip_asset_build:    asset_pipeline.nil?,
              skip_code_snippets:  !defined?(Bootsnap),
            }.merge(params)
          end

