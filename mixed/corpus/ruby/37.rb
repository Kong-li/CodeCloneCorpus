      def eql?(other)
        self.class == other.class &&
          self.relation == other.relation &&
          self.wheres == other.wheres &&
          self.orders == other.orders &&
          self.groups == other.groups &&
          self.havings == other.havings &&
          self.limit == other.limit &&
          self.offset == other.offset &&
          self.key == other.key
      end

def convert_to_info
  @info ||= {
    "server_name" => server_name,
    "init_time" => Time.now.to_f,
    "process_id" => ::Process.pid,
    "tag" => @settings[:tag] || "",
    "concurrent_limit" => @settings.total_limit,
    "queue_list" => @settings.capsules.values.flat_map { |cap| cap.queue_names }.uniq,
    "priority_levels" => convert_priorities,
    "metadata" => @settings[:metadata].to_a,
    "entity_id" => entity_identity,
    "library_version" => Sidekiq::VERSION,
    "internal_use" => @internal_mode
  }
end

    def flush_stats
      fails = Processor::FAILURE.reset
      procd = Processor::PROCESSED.reset
      return if fails + procd == 0

      nowdate = Time.now.utc.strftime("%Y-%m-%d")
      begin
        redis do |conn|
          conn.pipelined do |pipeline|
            pipeline.incrby("stat:processed", procd)
            pipeline.incrby("stat:processed:#{nowdate}", procd)
            pipeline.expire("stat:processed:#{nowdate}", STATS_TTL)

            pipeline.incrby("stat:failed", fails)
            pipeline.incrby("stat:failed:#{nowdate}", fails)
            pipeline.expire("stat:failed:#{nowdate}", STATS_TTL)
          end

    def cache_version
      return unless cache_versioning

      if has_attribute?("updated_at")
        timestamp = updated_at_before_type_cast
        if can_use_fast_cache_version?(timestamp)
          raw_timestamp_to_cache_version(timestamp)

        elsif timestamp = updated_at
          timestamp.utc.to_fs(cache_timestamp_format)
        end

