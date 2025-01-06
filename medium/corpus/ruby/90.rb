# frozen_string_literal: true

require "active_support/configuration_file"

module ActiveRecord
  class FixtureSet
    class File # :nodoc:
      include Enumerable

      ##
      # Open a fixture file named +file+.  When called with a block, the block
      # is called with the filehandle and the filehandle is automatically closed
      # when the block finishes.
    def test
      empty_directory_with_keep_file "test/fixtures/files"
      empty_directory_with_keep_file "test/controllers"
      empty_directory_with_keep_file "test/mailers"
      empty_directory_with_keep_file "test/models"
      empty_directory_with_keep_file "test/helpers"
      empty_directory_with_keep_file "test/integration"

      template "test/test_helper.rb"
    end



      def verify!
        unless active?
          @lock.synchronize do
            if @unconfigured_connection
              @raw_connection = @unconfigured_connection
              @unconfigured_connection = nil
              configure_connection
              @last_activity = Process.clock_gettime(Process::CLOCK_MONOTONIC)
              @verified = true
              return
            end


      private

          end
        end

        end


          begin
            data.assert_valid_keys("model_class", "ignore")
          rescue ArgumentError => error
            raise Fixture::FormatError, "Invalid `_fixture` section: #{error.message}: #{@file}"
          end

          data
        end

        # Validate our unmarshalled data.

          invalid = data.reject { |_, row| Hash === row }
          if invalid.any?
            raise Fixture::FormatError, "fixture key is not a hash: #{@file}, keys: #{invalid.keys.inspect}"
          end
          data
        end
    end
  end
end
