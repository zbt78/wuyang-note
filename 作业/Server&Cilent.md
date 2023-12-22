本程序使用 ServerSocket 类和 Socket 类实现服务器端和客户端程序。分为客户端和服务器端。在客户端输入密码连接服务器，输入密码时有三次机会，三次之后程序退出，视为非法用户。



Server：

```java
import java.io.*;
import java.net.*;

import static java.lang.System.exit;

public class Server {
    public static void main(String[] args) {
        try {
            //定义打开的port
            ServerSocket server = new ServerSocket(5050);
            Socket s = server.accept();
            BufferedReader in = new BufferedReader(new InputStreamReader(s.getInputStream()));
            PrintStream out = new PrintStream(s.getOutputStream());
            String str;
            //发送给客户端的内容
            out.println("Verifying Server!");
            out.println("Please input Password:");
            int i = 0;
            //自己定义的服务器密码
            String pwd = "wuyangOuO";
            while (i < 4) {
                //读取从客户端输入的密码
                str = in.readLine();
                //匹配成功，则登录成功
                if (pwd.equals(str)) {
                    out.println("Registration Successful!");
                    break;
                } else {
                    i++;
                    //如果失败总共有三次机会
                    if (i < 3) {
                        out.println("PassWord Wrong!");
                    } else {
                        out.println("Illegal User!");
                        break;
                    }
                }
            }
            in.close();
            out.close();
            s.close();
            server.close();
        } catch (IOException e) {
//            e.printStackTrace();

        }
    }
}
```

Client:

```java
import java.io.*;
import java.net.*;
import java.util.*;


public class Client {
    public static void main(String[] args) {
        try {
            //要与服务器的host和port保持一致
            Socket s = new Socket("127.0.0.1", 5050);
            BufferedReader in = new BufferedReader(new InputStreamReader(s.getInputStream()));
            PrintStream out = new PrintStream(s.getOutputStream());
            //从Server接收的消息
            String str = in.readLine();
            System.out.println(str);
            int i = 0;
            while (i < 4) {
                //为连接成功
                if (!"Verifying Server!".equals(str)) {
                    System.out.println("Server Wrong!");
                    break;
                } else {
                    String str2 = in.readLine();
                    System.out.println(str2);
                    if ("Registration Successful!".equals(str2)) {
                        break;
                    }
                    else if ("Illegal User!".equals(str2)) {
                        break;
                    }
                    else if ("PassWord Wrong!".equals(str2)) {
                        System.out.println("Please input Password:");
                    }
                    Scanner sc = new Scanner(System.in);
                    String str1 = sc.next();
                    //把密码发送给Server
                    out.println(str1);

                    i++;
                }
            }
            in.close();
            out.close();
            s.close();
        } catch (IOException e) {
//            e.printStackTrace();
            System.out.println("connect failed!");
        }
    }
}
```

结果：

- 三次都密码错误：

  ![image-20221023172434647](/Users/wuyangouo/Library/Application Support/typora-user-images/image-20221023172434647.png)

  ![image-20221023172503921](/Users/wuyangouo/Library/Application Support/typora-user-images/image-20221023172503921.png)

- 密码正确：
  ![image-20221023172544728](/Users/wuyangouo/Library/Application Support/typora-user-images/image-20221023172544728.png)
  ![image-20221023172614997](/Users/wuyangouo/Library/Application Support/typora-user-images/image-20221023172614997.png)

- Server未开启：

  ![image-20221023172709648](/Users/wuyangouo/Library/Application Support/typora-user-images/image-20221023172709648.png)