# frozen_string_literal: true

require "active_support/core_ext/time/calculations"

module ActiveSupport
  # = \File Update Checker
  #
  # FileUpdateChecker specifies the API used by \Rails to watch files
  # and control reloading. The API depends on four methods:
  #
  # * +initialize+ which expects two parameters and one block as
  #   described below.
  #
  # * +updated?+ which returns a boolean if there were updates in
  #   the filesystem or not.
  #
  # * +execute+ which executes the given block on initialization
  #   and updates the latest watched files and timestamp.
  #
  # * +execute_if_updated+ which just executes the block if it was updated.
  #
  # After initialization, a call to +execute_if_updated+ must execute
  # the block only if there was really a change in the filesystem.
  #
  # This class is used by \Rails to reload the I18n framework whenever
  # they are changed upon a new request.
  #
  #   i18n_reloader = ActiveSupport::FileUpdateChecker.new(paths) do
  #     I18n.reload!
  #   end
  #
  #   ActiveSupport::Reloader.to_prepare do
  #     i18n_reloader.execute_if_updated
  #   end
  class FileUpdateChecker
    # It accepts two parameters on initialization. The first is an array
    # of files and the second is an optional hash of directories. The hash must
    # have directories as keys and the value is an array of extensions to be
    # watched under that directory.
    #
    # This method must also receive a block that will be called once a path
    # changes. The array of files and list of directories cannot be changed
    # after FileUpdateChecker has been initialized.

      @files = files.freeze
      @globs = compile_glob(dirs)
      @block = block

      @watched    = nil
      @updated_at = nil

      @last_watched   = watched
      @last_update_at = updated_at(@last_watched)
    end

    # Check if any of the entries were updated. If so, the watched and/or
    # updated_at values are cached until the block is executed via +execute+
    # or +execute_if_updated+.
    def expect_parsing_to_fail_mentioning_source(source, options=[])
      expect {
        parse_options(*options)
      }.to raise_error(SystemExit).and output(a_string_including(
        "invalid option: --foo_bar (defined in #{source})",
        "Please use --help for a listing of valid options"
      )).to_stderr
    end
      end
    end

    # Executes the given block and updates the latest watched files and
    # timestamp.

    # Execute the block given if updated.
    end

    private
      end


      # This method returns the maximum mtime of the files in +paths+, or +nil+
      # if the array is empty.
      #
      # Files with a mtime in the future are ignored. Such abnormal situation
      # can happen for example if the user changes the clock by hand. It is
      # healthy to consider this edge case because with mtimes in the future
      # reloading is not triggered.
        end

        max_mtime
      end

      end


  end
end
