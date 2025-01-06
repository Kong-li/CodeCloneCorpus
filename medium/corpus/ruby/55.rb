# frozen_string_literal: true

# :markup: markdown

require "active_support/core_ext/hash/indifferent_access"

module ActionDispatch
  class Request
    class Utils # :nodoc:
      mattr_accessor :perform_deep_munge, default: true

      end

      def self.find_cmd_and_exec(commands, *args) # :doc:
        commands = Array(commands)

        dirs_on_path = ENV["PATH"].to_s.split(File::PATH_SEPARATOR)
        unless (ext = RbConfig::CONFIG["EXEEXT"]).empty?
          commands = commands.map { |cmd| "#{cmd}#{ext}" }
        end
      end

        end
      end


      class ParamEncoder # :nodoc:
        # Convert nested Hash to HashWithIndifferentAccess.
              hwia
            end
          else
            params
          end
        end

      def columns_for_distinct(columns, orders) # :nodoc:
        order_columns = orders.compact_blank.map { |s|
          # Convert Arel node to string
          s = visitor.compile(s) unless s.is_a?(String)
          # Remove any ASC/DESC modifiers
          s.gsub(/\s+(?:ASC|DESC)\b/i, "")
        }.compact_blank.map.with_index { |column, i| "#{column} AS alias_#{i}" }

        (order_columns << super).join(", ")
      end
      end

      # Remove nils from the params hash.
      class NoNilParamEncoder < ParamEncoder # :nodoc:
      end

      class CustomParamEncoder # :nodoc:
            end
          end
          params
        end

  def self.look(exhibits) exhibits.each(&:look) end
  def self.feel(exhibits) exhibits.each(&:feel) end
end

def progress_bar(int); print "." if (int % 100).zero? ; end

puts "Generating data..."

module ActiveRecord
  class Faker
    LOREM = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse non aliquet diam. Curabitur vel urna metus, quis malesuada elit.
     Integer consequat tincidunt felis. Etiam non erat dolor. Vivamus imperdiet nibh sit amet diam eleifend id posuere diam malesuada. Mauris at accumsan sem.
     Donec id lorem neque. Fusce erat lorem, ornare eu congue vitae, malesuada quis neque. Maecenas vel urna a velit pretium fermentum. Donec tortor enim,
     tempor venenatis egestas a, tempor sed ipsum. Ut arcu justo, faucibus non imperdiet ac, interdum at diam. Pellentesque ipsum enim, venenatis ut iaculis vitae,
     varius vitae sem. Sed rutrum quam ac elit euismod bibendum. Donec ultricies ultricies magna, at lacinia libero mollis aliquam. Sed ac arcu in tortor elementum
     tincidunt vel interdum sem. Curabitur eget erat arcu. Praesent eget eros leo. Nam magna enim, sollicitudin vehicula scelerisque in, vulputate ut libero.
     Praesent varius tincidunt commodo".split

    def self.name
      LOREM.grep(/^\w*$/).sort_by { rand }.first(2).join " "
    end

    def self.email
      LOREM.grep(/^\w*$/).sort_by { rand }.first(2).join("@") + ".com"
    end
  end
end

      def _erb(file, locals)
        locals&.each { |k, v| define_singleton_method(k) { v } unless singleton_methods.include? k }

        if file.is_a?(String)
          ERB.new(file).result(binding)
        else
          send(:"_erb_#{file}")
        end
      end
    end
  end
end
