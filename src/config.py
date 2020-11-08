from time import sleep
import configparser

default={
    "general" : {"target": "BARRIER"},
    "sensors" : {
        "cornerPos": "31, 43, 196, 208",
        "gridColums": "15",
        "gridRows": "15",
        "reaction": "1",
        "bufferSize": "6",
        "showGrid": "false"},
    "gyroscope" : {
        "bufferSize": "3",
        "digitMinArea": "0.00040",
        "digitMaxArea": "0.0019"}
}

def open_config(appName):
    try:
        open(f'{homeDir}{appName}.cfg', encoding='utf-8')
    except IOError:
        open(f'{homeDir}{appName}.cfg', 'tw', encoding='utf-8')

    cfg = configparser.RawConfigParser(comment_prefixes=('#', ';', '//'), allow_no_value=True)
    try:
        cfg.read(f'{homeDir}{appName}.cfg')
    except Exception as e:
        print(f"Error to read configuration file: {e}")
        sleep(3)
        raise SystemExit(1)
    return cfg

def write_section(section, params):
    def lowcaseMe(val):
        return val.lower()

    def configWrite():
        with open(f'{homeDir}{appName}.cfg', "w") as configFile:
            cfg.write(configFile)

    cfg.optionxform = str  # позволяет записать параметр сохранив регистр
    cfg.add_section(section)
    for val in params:
        cfg.set(section, val, params[val])
        cfg.optionxform = lowcaseMe  # возращаем предопределённый метод назад
    configWrite()
    return True

def check_sections(cfg):
    try:
        new = False
        for key in default:
            if not cfg.has_section(key):
                new = write_section(key, default[key])

        if new is True:
            print("WARNING: Были созданы новые секции в файле конфигурации. "
                  "Для их действия запустите приложение заново.")
            sleep(3)
            raise SystemExit(1)
    except Exception as e:
        print(f"ERROR: Не удалось создать файл конфигурации: {e}")
        sleep(3)
        raise SystemExit(1)

# исполльзую default как каркас
def load_config(cfg):
    def get_value(section, param, type=0):
        if type == 1:
            val = cfg.getboolean(section, param)
        elif type == 2:
            val = cfg.getint(section, param)
        elif type == 3:
            val = cfg.getfloat(section, param)
        else:
            val = cfg.get(section, param)
        
        settings[section][param] = val
    
    check_sections(cfg)
    settings = dict(default)
    try:
        get_value("general", "target")
        get_value("sensors", "cornerPos")
        get_value("sensors", "gridColums", 2)
        get_value("sensors", "gridRows", 2)
        get_value("sensors", "reaction", 2)
        get_value("sensors", "bufferSize", 2)
        get_value("sensors", "showGrid", 1)
        get_value("gyroscope", "bufferSize", 2)
        get_value("gyroscope", "digitMinArea", 3)
        get_value("gyroscope", "digitMaxArea", 3)

        settings['sensors']['cornerPos'] = [int(i.strip()) for i in settings['sensors']['cornerPos'].split(',')]
        settings['sensors']['sensorsPos'] = []
        
        return settings
    except Exception as e:
        print(f"WARNING: Check parameters: {e}")
        sleep(3)
        raise SystemExit(1)
    
if __name__ == "__main__":
    print('Use xBot to start bot')
    sleep(3)

from __main__ import appName, homeDir

cfg = open_config(appName)
settings = load_config(cfg)
