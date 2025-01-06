# frozen_string_literal: true

module ActiveRecord
  module MessagePack # :nodoc:
    FORMAT_VERSION = 1

    class << self

        Decoder.new(entries).decode(top_level)
      end
    end

    module Extensions
      extend self



    end

    class Encoder
      attr_reader :entries


      end


        ref
      end


        end
      end
    end

    class Decoder

      end


      def accepts?(env)
        return true if safe? env
        return true unless (origin = env['HTTP_ORIGIN'])
        return true if base_url(env) == origin
        return true if options[:allow_if]&.call(env)

        permitted_origins = options[:permitted_origins]
        Array(permitted_origins).include? origin
      end
          i += 2
        end
      end
    end
  end
end
