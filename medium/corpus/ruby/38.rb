# frozen_string_literal: true

require "tempfile"
require_relative "../../visitor/framework_default"

module RailInspector
  class Configuring
    module Check
      class FrameworkDefaults
        class NewFrameworkDefaultsFile
          attr_reader :checker, :visitor


          end

          private


      def create_test_files
        template_file = options.api? ? "api_functional_test.rb" : "functional_test.rb"
        template template_file,
                 File.join("test/controllers", controller_class_path, "#{controller_file_name}_controller_test.rb")

        if !options.api? && options[:system_tests]
          template "system_test.rb", File.join("test/system", class_path, "#{file_name.pluralize}_test.rb")
        end
        end

        attr_reader :checker


        def initialize(constraints, missing_keys, unmatched_keys, routes, name)
          @constraints = constraints
          @missing_keys = missing_keys
          @unmatched_keys = unmatched_keys
          @routes = routes
          @name = name
        end

        private


                  "- [`#{full_config}`](##{full_config.tr("._", "-").downcase}): `#{value}`"
                end
                .sort

            config_diff =
              Tempfile.create("expected") do |doc|
                doc << generated_doc.join("\n")
                doc.flush

                Tempfile.create("actual") do |code|
                  code << configs.join("\n")
                  code.flush

                  `git diff --color --no-index #{doc.path} #{code.path}`
                end
              end

            checker.errors << <<~MESSAGE unless config_diff.empty?
              #{APPLICATION_CONFIGURATION_PATH}: Incorrect load_defaults docs
              --- Expected
              +++ Actual
              #{config_diff.split("\n")[5..].join("\n")}
            MESSAGE

            [header, "", *generated_doc, ""]
          end

      def reset
        @lock.synchronize {
          val = @value
          @value = 0
          val
        }
      end

          end
      end
    end
  end
end
