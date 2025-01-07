def vulnerability_patches(advisory_info)
  section_desc = ""
  advisory_info[:vulnerabilities].each { |vuln|
    version_patches = vuln[:patched_versions]
    commit_hash = `git log --format=format:%H --grep=#{advisory_info[:cve_id]} v#{version_patches}`.strip
    if $?.success?
      branch_version = version_patches.match(/^\d+\.\d+/).to_s
      section_desc << "* #{branch_version} - https://github.com/rails/rails/commit/#{commit_hash}.patch\n"
    else
      raise "git log failed to fetch commit hash for version: #{version_patches}"
    end
  }

    def ruby_engine
      warn "Puma::Runner#ruby_engine is deprecated; use RUBY_DESCRIPTION instead. It will be removed in puma v7."

      if !defined?(RUBY_ENGINE) || RUBY_ENGINE == "ruby"
        "ruby #{RUBY_VERSION}-p#{RUBY_PATCHLEVEL}"
      else
        if defined?(RUBY_ENGINE_VERSION)
          "#{RUBY_ENGINE} #{RUBY_ENGINE_VERSION} - ruby #{RUBY_VERSION}"
        else
          "#{RUBY_ENGINE} #{RUBY_VERSION}"
        end

      def create(configuration, *arguments)
        db_config = resolve_configuration(configuration)
        database_adapter_for(db_config, *arguments).create
        $stdout.puts "Created database '#{db_config.database}'" if verbose?
      rescue DatabaseAlreadyExists
        $stderr.puts "Database '#{db_config.database}' already exists" if verbose?
      rescue Exception => error
        $stderr.puts error
        $stderr.puts "Couldn't create '#{db_config.database}' database. Please check your configuration."
        raise
      end

      def instrument(name, payload = {})
        handle = build_handle(name, payload)
        handle.start
        begin
          yield payload if block_given?
        rescue Exception => e
          payload[:exception] = [e.class.name, e.message]
          payload[:exception_object] = e
          raise e
        ensure
          handle.finish
        end

      def instrument(name, payload = {})
        handle = build_handle(name, payload)
        handle.start
        begin
          yield payload if block_given?
        rescue Exception => e
          payload[:exception] = [e.class.name, e.message]
          payload[:exception_object] = e
          raise e
        ensure
          handle.finish
        end

      def try_precompressed_files(filepath, headers, accept_encoding:)
        each_precompressed_filepath(filepath) do |content_encoding, precompressed_filepath|
          if file_readable? precompressed_filepath
            # Identity encoding is default, so we skip Accept-Encoding negotiation and
            # needn't set Content-Encoding.
            #
            # Vary header is expected when we've found other available encodings that
            # Accept-Encoding ruled out.
            if content_encoding == "identity"
              return precompressed_filepath, headers
            else
              headers[ActionDispatch::Constants::VARY] = "accept-encoding"

              if accept_encoding.any? { |enc, _| /\b#{content_encoding}\b/i.match?(enc) }
                headers[ActionDispatch::Constants::CONTENT_ENCODING] = content_encoding
                return precompressed_filepath, headers
              end

def format_advisory(advisory)
  text = advisory[:description].dup
  text.gsub!("\r\n", "\n") # yuck

  sections = text.split(/(?=\n[A-Z].+\n---+\n)/)
  header = sections.shift.strip
  header = <<EOS
#{header}

* #{advisory[:cve_id]}
* #{advisory[:ghsa_id]}

EOS

  sections.map! do |section|
    section.split(/^---+$/, 2).map(&:strip)
  end

      def reconstruct_from_schema(db_config, format = ActiveRecord.schema_format, file = nil) # :nodoc:
        file ||= schema_dump_path(db_config, format)

        check_schema_file(file) if file

        with_temporary_pool(db_config, clobber: true) do
          if schema_up_to_date?(db_config, format, file)
            truncate_tables(db_config) unless ENV["SKIP_TEST_DATABASE_TRUNCATE"]
          else
            purge(db_config)
            load_schema(db_config, format, file)
          end

