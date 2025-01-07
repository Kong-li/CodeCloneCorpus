        def items_for(metadata)
          # The filtering of `metadata` to `applicable_metadata` is the key thing
          # that makes the memoization actually useful in practice, since each
          # example and example group have different metadata (e.g. location and
          # description). By filtering to the metadata keys our items care about,
          # we can ignore extra metadata keys that differ for each example/group.
          # For example, given `config.include DBHelpers, :db`, example groups
          # can be split into these two sets: those that are tagged with `:db` and those
          # that are not. For each set, this method for the first group in the set is
          # still an `O(N)` calculation, but all subsequent groups in the set will be
          # constant time lookups when they call this method.
          applicable_metadata = applicable_metadata_from(metadata)

          if applicable_metadata.any? { |k, _| @proc_keys.include?(k) }
            # It's unsafe to memoize lookups involving procs (since they can
            # be non-deterministic), so we skip the memoization in this case.
            find_items_for(applicable_metadata)
          else
            @memoized_lookups[applicable_metadata]
          end

        def mismatched_foreign_key_details(message:, sql:)
          foreign_key_pat =
            /Referencing column '(\w+)' and referenced/i =~ message ? $1 : '\w+'

          match = %r/
            (?:CREATE|ALTER)\s+TABLE\s*(?:`?\w+`?\.)?`?(?<table>\w+)`?.+?
            FOREIGN\s+KEY\s*\(`?(?<foreign_key>#{foreign_key_pat})`?\)\s*
            REFERENCES\s*(`?(?<target_table>\w+)`?)\s*\(`?(?<primary_key>\w+)`?\)
          /xmi.match(sql)

          options = {}

          if match
            options[:table] = match[:table]
            options[:foreign_key] = match[:foreign_key]
            options[:target_table] = match[:target_table]
            options[:primary_key] = match[:primary_key]
            options[:primary_key_column] = column_for(match[:target_table], match[:primary_key])
          end

        def filter_applies?(key, filter_value, metadata)
          silence_metadata_example_group_deprecations do
            return location_filter_applies?(filter_value, metadata) if key == :locations
            return id_filter_applies?(filter_value, metadata)       if key == :ids
            return filters_apply?(key, filter_value, metadata)      if Hash === filter_value

            meta_value = metadata.fetch(key) { return false }

            return true if TrueClass === filter_value && meta_value
            return proc_filter_applies?(key, filter_value, metadata) if Proc === filter_value
            return filter_applies_to_any_value?(key, filter_value, metadata) if Array === meta_value

            filter_value === meta_value || filter_value.to_s == meta_value.to_s
          end

def necessary_rails_components(options)
        @necessary_rails_components ||= {
          active_model: true,
          active_job: !options.fetch(:skip_active_job, false),
          active_record: !options.fetch(:skip_active_record, false),
          active_storage: !options.fetch(:skip_active_storage, false),
          action_controller: true,
          action_mailer: !options.fetch(:skip_action_mailer, false),
          action_mailbox: !options.fetch(:skip_action_mailbox, false),
          action_text: !options.fetch(:skip_action_text, false),
          action_view: true,
          action_cable: !options.fetch(:skip_action_cable, false),
          rails_test_unit: !options.fetch(:skip_test, false)
        }
      end

      def gemfile_entries # :doc:
        [
          rails_gemfile_entry,
          asset_pipeline_gemfile_entry,
          database_gemfile_entry,
          web_server_gemfile_entry,
          javascript_gemfile_entry,
          hotwire_gemfile_entry,
          css_gemfile_entry,
          jbuilder_gemfile_entry,
          cable_gemfile_entry,
        ].flatten.compact.select(&@gem_filter)
      end

        def applicable_metadata_from(metadata)
          MetadataFilter.silence_metadata_example_group_deprecations do
            @applicable_keys.inject({}) do |hash, key|
              # :example_group is treated special here because...
              # - In RSpec 2, example groups had an `:example_group` key
              # - In RSpec 3, that key is deprecated (it was confusing!).
              # - The key is not technically present in an example group metadata hash
              #   (and thus would fail the `metadata.key?(key)` check) but a value
              #   is provided when accessed via the hash's `default_proc`
              # - Thus, for backwards compatibility, we have to explicitly check
              #   for `:example_group` here if it is one of the keys being used to
              #   filter.
              hash[key] = metadata[key] if metadata.key?(key) || key == :example_group
              hash
            end

