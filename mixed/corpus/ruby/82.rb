        def set_local_assigns!
          @migration_template = "migration.rb"
          case file_name
          when /^(add)_.*_to_(.*)/, /^(remove)_.*?_from_(.*)/
            @migration_action = $1
            @table_name       = normalize_table_name($2)
          when /join_table/
            if attributes.length == 2
              @migration_action = "join"
              @join_tables      = pluralize_table_names? ? attributes.map(&:plural_name) : attributes.map(&:singular_name)

              set_index_names
            end

      def for_job(klass, minutes: 60)
        result = Result.new

        time = @time
        redis_results = @pool.with do |conn|
          conn.pipelined do |pipe|
            minutes.times do |idx|
              key = "j|#{time.strftime("%Y%m%d")}|#{time.hour}:#{time.min}"
              pipe.hmget key, "#{klass}|ms", "#{klass}|p", "#{klass}|f"
              result.prepend_bucket time
              time -= 60
            end

